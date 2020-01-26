from collections import deque

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models.detection.transform import resize_boxes

from tracktor.training_set_generation import replicate_and_randomize_boxes
from tracktor.utils import clip_boxes
from tracktor.visualization import plot_bounding_boxes, VisdomLinePlotter
import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps, im_info,
                 transformed_image_size, box_roi_pool=None):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.frames_since_active = 1
        self.last_v = torch.Tensor([])
        self.transformed_image_size = transformed_image_size
        self.gt_id = None
        self.im_info = im_info
        self.box_predictor_classification = None
        self.box_head_classification = None
        self.box_predictor_regression = None
        self.box_head_regression = None
        self.scale = self.im_info[0] / self.transformed_image_size[0][0]
        self.plotter = VisdomLinePlotter(env_name='training')
        self.checkpoints = dict()
        self.training_set = IndividualDataset(self.id)
        self.box_roi_pool = box_roi_pool

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())

    # TODO is displacement of roi helpful? Kinda like dropout as specific features might not be in the ROI anymore
    # TODO only take negative examples that are close to positive example --> Makes training easier.
    # TODO try lower learning rate and not to overfit --> best behaviour of 6 was when 0 track still had high score.
    def generate_training_set_regression(self, gt_pos, max_displacement, batch_size=8, plot=False, plot_args=None):
        gt_pos = gt_pos.to(device)
        random_displaced_bboxes = replicate_and_randomize_boxes(gt_pos,
                                                                batch_size=batch_size,
                                                                max_displacement=max_displacement).to(device)

        training_boxes = clip_boxes(random_displaced_bboxes, self.im_info)

        if plot and plot_args:
            plot_bounding_boxes(self.im_info, gt_pos, plot_args[0], training_boxes.numpy(), plot_args[1], plot_args[2])

        return training_boxes

    def generate_training_set_classification(self, batch_size, additional_dets, fpn_features, shuffle=False):
        num_positive_examples = int(batch_size / 2)
        positive_examples = self.generate_training_set_regression(self.pos,
                                                                  0.0,
                                                                  batch_size=num_positive_examples).to(device)
        positive_examples = clip_boxes(positive_examples, self.im_info)
        # positive_examples = self.pos.repeat(num_positive_examples, 1)
        positive_examples = torch.cat((positive_examples, torch.ones([num_positive_examples, 1]).to(device)), dim=1)
        boxes = positive_examples
        if additional_dets.size(0) != 0:
            standard_batch_size_negative_example = int(np.floor(num_positive_examples / len(additional_dets)))
            offset = num_positive_examples - (standard_batch_size_negative_example * additional_dets.size(0))
            for i in range(additional_dets.size(0)):
                num_negative_example = standard_batch_size_negative_example
                if offset != 0:
                    num_negative_example += 1
                    offset -= 1
                if num_negative_example == 0:
                    break
                negative_example = self.generate_training_set_regression(additional_dets[i].view(1, -1),
                                                                         0.0,
                                                                         batch_size=num_negative_example).to(device)
                negative_example = clip_boxes(negative_example, self.im_info)
                # negative_example = additional_dets[i].view(1, -1).repeat(num_negative_example, 1)
                negative_example_and_label = torch.cat((negative_example, torch.zeros([num_negative_example, 1]).to(device)), dim=1)
                boxes = torch.cat((boxes, negative_example_and_label)).to(device)
        if shuffle:
            boxes = boxes[torch.randperm(boxes.size(0))]
        boxes_resized = resize_boxes(boxes[:, 0:4], self.im_info, self.transformed_image_size[0])
        proposals = [boxes_resized]
        with torch.no_grad():
            roi_pool_feat = self.box_roi_pool(fpn_features, proposals, self.im_info).to(device)

        return {'features': roi_pool_feat, 'boxes': boxes[:, 0:4], 'scores': boxes[:, 4]}

    def update_training_set_classification(self, batch_size, additional_dets, fpn_features,
                                           include_previous_frames=False, shuffle=False, replacement_probability=0.5):
        training_set_dict = self.generate_training_set_classification(batch_size, additional_dets, fpn_features, shuffle=shuffle)
        if include_previous_frames and self.training_set['features'] is not None:
            weights = torch.tensor([1 / batch_size]).repeat(int(batch_size))
            indices_replaced_by_current_frame_features = torch.multinomial(weights, int(batch_size * replacement_probability))
            self.training_set['features'][indices_replaced_by_current_frame_features] = training_set_dict['features'][indices_replaced_by_current_frame_features]
            self.training_set['boxes'][indices_replaced_by_current_frame_features] = training_set_dict['boxes'][
                indices_replaced_by_current_frame_features]

        else:
            self.training_set = training_set_dict

    def generate_validation_set_classfication(self, batch_size, additional_dets, fpn_features, shuffle=False):
        return self.generate_training_set_classification(batch_size, additional_dets, fpn_features, shuffle=shuffle)

    def forward_pass_for_classifier_training(self, features, scores, eval=False, return_scores=False):
        if eval:
            self.box_predictor_classification.eval()
            self.box_head_classification.eval()
#        boxes_resized = resize_boxes(boxes[:, 0:4], self.im_info, self.transformed_image_size[0])
#        proposals = [boxes_resized]
#        with torch.no_grad():
#            roi_pool_feat = box_roi_pool(fpn_features, proposals, self.im_info)
        feat = self.box_head_classification(features)
        class_logits, _ = self.box_predictor_classification(feat)
        if return_scores:
            pred_scores = F.softmax(class_logits, -1)
            if eval:
                self.box_predictor_classification.train()
                self.box_head_classification.train()
            return pred_scores[:, 1:].squeeze(dim=1).detach()
        loss = F.cross_entropy(class_logits, scores.long())
        if eval:
            self.box_predictor_classification.train()
            self.box_head_classification.train()
        return loss

    def finetune_classification(self, finetuning_config, box_head_classification, box_predictor_classification,
                                early_stopping=False):
        self.box_head_classification = box_head_classification
        self.box_predictor_classification = box_predictor_classification

        self.box_predictor_classification.train()
        self.box_head_classification.train()
        optimizer = torch.optim.Adam(
            list(self.box_predictor_classification.parameters()) + list(self.box_head_classification.parameters()), lr=float(finetuning_config["learning_rate"]) )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=finetuning_config['gamma'])
        dataloader = torch.utils.data.DataLoader(self.training_set, batch_size=128)

        # if finetuning_config["validate"]:# and additional_dets is not None:
        #     if not self.plotter:
        #         self.plotter = VisdomLinePlotter(env_name='training')
        #     additional_dets = additional_dets[:-1]
        #     val_dets = additional_dets[-1:]
        #     validation_set = self.generate_validation_set_classfication(float(int(finetuning_config["batch_size_val"])),
          #                                                               val_dets, fpn_features)
        print("Finetuning track {}".format(self.id))
        for i in range(int(finetuning_config["iterations"])):
            for i_sample, sample_batch in enumerate(dataloader):

                optimizer.zero_grad()
                loss = self.forward_pass_for_classifier_training(sample_batch['features'], sample_batch['scores'], eval=False)

                if early_stopping or finetuning_config["validate"]:
                    scores = self.forward_pass_for_classifier_training(sample_batch['features'], sample_batch['scores'], return_scores=True, eval=True)

                if finetuning_config["validate"]:
                    self.plotter.plot('loss', 'positive', 'Class Loss Evaluation Track {}'.format(self.id), i, scores[0].cpu().numpy(), is_target=True)
                    for sample in range(16, 32):
                        self.plotter.plot('loss', 'negative {}'.format(sample), 'Class Loss Evaluation Track {}'.format(self.id), i, scores[sample].cpu().numpy())

                    if early_stopping and scores[0] - torch.max(scores[16:]) > 0.8:
                        print('Stopping because difference between positive score and maximum negative score is {}'.format(scores[0] - torch.max(scores[16:])))
                        break

                loss.backward()
                optimizer.step()
                scheduler.step()

        self.box_predictor_classification.eval()
        self.box_head_classification.eval()

        # dets = torch.cat((self.pos, additional_dets))
        # print(self.forward_pass(dets, box_roi_pool, fpn_features, scores=True))

class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id):
        self.id = id
        self.features = torch.tensor([])
        self.boxes = torch.tensor([])
        self.scores = torch.tensor([])

    def append_samples(self, training_set_dict):
        self.features = torch.cat((self.features, training_set_dict['features']))
        self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        self.scores = torch.cat((self.scores, training_set_dict['scores']))

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}

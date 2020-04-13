from collections import deque

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models.detection.transform import resize_boxes

from tracktor.training_set_generation import replicate_and_randomize_boxes
from tracktor.utils import clip_boxes
from tracktor.visualization import plot_bounding_boxes, VisdomLinePlotter
from tracktor.live_dataset import IndividualDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps, im_info,
                 transformed_image_size, batch_size, keep_frames, plot=False, box_roi_pool=None):
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
        if plot:
            self.plotter = VisdomLinePlotter(env_name='training')
        self.checkpoints = dict()
        self.training_set = IndividualDataset(self.id, batch_size, keep_frames)
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
        if additional_dets.size(0) == 0:
            print("Adding dummy bbox as negative example")
            additional_dets = torch.Tensor([1892.4128,  547.1268, 1919.0000,  629.0942]).unsqueeze(0)

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
                                           include_previous_frames=False, shuffle=False):
        training_set_dict = self.generate_training_set_classification(batch_size, additional_dets, fpn_features, shuffle=shuffle)

        if not include_previous_frames:
            self.training_set = IndividualDataset(self.id, batch_size)
        self.training_set.append_samples(training_set_dict)

    def forward_pass_for_classifier_training(self, features, scores, eval=False, return_scores=False):
        if eval:
            self.box_predictor_classification.eval()
            self.box_head_classification.eval()
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
        self.training_set.post_process()

        self.box_head_classification = box_head_classification
        self.box_predictor_classification = box_predictor_classification

        self.box_predictor_classification.train()
        self.box_head_classification.train()
        optimizer = torch.optim.Adam(
            list(self.box_predictor_classification.parameters()) + list(self.box_head_classification.parameters()), lr=float(finetuning_config["learning_rate"]) )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=finetuning_config['gamma'])

        training_set, val_set = self.training_set.get_training_set()
        dataloader_train = torch.utils.data.DataLoader(training_set, batch_size=256)
        dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=256)

        for i in range(int(finetuning_config["iterations"])):
            for i_sample, sample_batch in enumerate(dataloader_train):

                optimizer.zero_grad()
                loss = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                                 sample_batch['scores'], eval=False)

                loss.backward()
                optimizer.step()
                scheduler.step()
            if finetuning_config["validate"] or finetuning_config["plot_training_curves"]:
                positive_scores = self.forward_pass_for_classifier_training(sample_batch['features'][sample_batch['scores']==1], sample_batch['scores'], return_scores=True, eval=True)
                negative_scores = self.forward_pass_for_classifier_training(sample_batch['features'][sample_batch['scores']==0], sample_batch['scores'], return_scores=True, eval=True)

            if early_stopping:
                positive_scores = torch.Tensor([]).to(device)
                negative_scores = torch.Tensor([]).to(device)
                for val_batch_idx, val_batch in enumerate(dataloader_val):
                    pos_scores_batch = self.forward_pass_for_classifier_training(
                        val_batch['features'][val_batch['scores'] == 1], val_batch['scores'],
                        return_scores=True, eval=True)
                    positive_scores = torch.cat((positive_scores, pos_scores_batch))
                    neg_scores_batch = self.forward_pass_for_classifier_training(
                        val_batch['features'][val_batch['scores'] == 0], val_batch['scores'],
                        return_scores=True, eval=True)
                    negative_scores = torch.cat((negative_scores, neg_scores_batch))

            if finetuning_config["plot_training_curves"]:
                positive_scores = positive_scores[:10]
                negative_scores = negative_scores[:10]
                for sample_idx, score in enumerate(positive_scores):
                    self.plotter.plot('score', 'positive {}'.format(sample_idx),
                                 'Scores Evaluation Classifier for Track {}'.format(self.id),
                                 i, score.cpu().numpy(), train_positive=True)  # dark red
                for sample_idx, score in enumerate(negative_scores):
                    self.plotter.plot('score', 'negative {}'.format(sample_idx),
                                 'Scores Evaluation Classifier for Track {}'.format(self.id),
                                 i, score.cpu().numpy())

            if early_stopping and torch.min(positive_scores) > 0.99 and torch.min(positive_scores) - torch.max(negative_scores) > 0.99:
                print(f"Stopping early after {i+1} iterations. max pos: {torch.min(positive_scores)}, max neg: {torch.max(negative_scores)}")
                break


        self.box_predictor_classification.eval()
        self.box_head_classification.eval()

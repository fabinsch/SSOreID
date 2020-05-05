from collections import deque, defaultdict


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
        self.checkpoints = dict()
        self.box_roi_pool = box_roi_pool
        self.training_set = IndividualDataset(self.id, batch_size, keep_frames)
        self.skipped_for_train = 0

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
        boxes = clip_boxes(self.pos, self.im_info)
        boxes = torch.cat((boxes, torch.ones([1, 1]).to(device)), dim=1)

        if additional_dets.size(0) == 0:
            print("Adding dummy bbox as negative example")
            additional_dets = torch.Tensor([1892.4128,  547.1268, 1919.0000,  629.0942]).to(device).unsqueeze(0)

        for i in range(additional_dets.size(0)):
            negative_example = clip_boxes(additional_dets[i].view(1, -1), self.im_info)
            negative_example_and_label = torch.cat((negative_example, torch.zeros([1, 1]).to(device)), dim=1)
            boxes = torch.cat((boxes, negative_example_and_label)).to(device)

        # if shuffle:
        #     boxes = boxes[torch.randperm(boxes.size(0))]
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

    def add_classifier(self, box_head_classification, box_predictor_classification):
        self.box_head_classification_debug = box_head_classification
        self.box_predictor_classification_debug = box_predictor_classification


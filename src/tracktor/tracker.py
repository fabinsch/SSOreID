from collections import deque, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import cv2
import pandas as pd

from tracktor.training_set_generation import replicate_and_randomize_boxes
from tracktor.visualization import plot_compare_bounding_boxes, VisdomLinePlotter, plot_bounding_boxes, \
    parse_ground_truth
from tracktor.utils import clip_boxes
from tracktor.utils import bbox_overlaps, warp_pos, get_center, get_height, get_width, make_pos

from torchvision.ops.boxes import clip_boxes_to_image, nms, box_iou
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.transform import resize_boxes

import matplotlib
#if not torch.cuda.is_available():
#    matplotlib.use('TkAgg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg):
        self.obj_detect = obj_detect
        self.reid_network = reid_network
        self.detection_person_thresh = tracker_cfg['detection_person_thresh']
        self.regression_person_thresh = tracker_cfg['regression_person_thresh']
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']
        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']
        self.finetuning_config = tracker_cfg['finetuning']
        if self.finetuning_config["enabled"]:
            self.bbox_predictor_weights = self.obj_detect.roi_heads.box_predictor.state_dict()
            self.bbox_head_weights = self.obj_detect.roi_heads.box_head.state_dict()

        self.plotter = VisdomLinePlotter(env_name='person_scores', xlabel="Frames")
        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features, image):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        old_tracks = self.get_pos()
        for i in range(num_new):
            track = Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i,
                          new_det_features[i].view(1, -1), self.inactive_patience, self.max_features_num,
                          self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1,
                          image.size()[1:3], self.obj_detect.image_size)

            if self.finetuning_config["enabled"]:
                other_pedestrians_bboxes = torch.cat((new_det_pos[:i], new_det_pos[i+1:], old_tracks))
                box_head_copy = self.get_box_head()
                box_predictor_copy = self.get_box_predictor()
                track.finetune_detector(self.obj_detect.roi_heads.box_roi_pool,
                                        self.obj_detect.fpn_features,
                                        self.finetuning_config,
                                        box_head_copy,
                                        box_predictor_copy,
                                        additional_dets=other_pedestrians_bboxes)
            self.tracks.append(track)

        self.track_num += num_new

    def get_box_predictor(self):
        box_predictor = FastRCNNPredictor(1024, 2).to(device)
        box_predictor.load_state_dict(self.bbox_predictor_weights)
        return box_predictor

    def get_box_head(self):
        box_head =  TwoMLPHead(self.obj_detect.backbone.out_channels *
                                   self.obj_detect.roi_heads.box_roi_pool.output_size[0] ** 2,
                                   representation_size=1024).to(device)
        box_head.load_state_dict(self.bbox_head_weights)
        return box_head

    def regress_tracks(self, blob, plot_compare=False, frame=None):
        """Regress the position of the tracks and also checks their scores."""
        if self.finetuning_config["enabled"]:
            scores = []
            pos = []
            box_pred_id_6 = [(track.box_head, track.box_predictor) for track in self.tracks if track.id == 6]
            for track in self.tracks:
                # Regress with finetuned bbox head for each track
                assert track.box_head is not None
                assert track.box_predictor is not None

                box, score = self.obj_detect.predict_boxes(track.pos,
                                                           box_head=track.box_head,
                                                           box_predictor=track.box_predictor)

                if plot_compare:
                    box_no_finetune, score_no_finetune = self.obj_detect.predict_boxes(track.pos)
                    plot_compare_bounding_boxes(box, box_no_finetune, blob['img'])
                scores.append(score)
                bbox = clip_boxes_to_image(box, blob['img'].shape[-2:])
                pos.append(bbox)
                if box_pred_id_6:
                    _, score_plot = self.obj_detect.predict_boxes(track.pos,
                                                                  box_head=box_pred_id_6[0][0],
                                                                  box_predictor=box_pred_id_6[0][1])
                    self.plotter.plot('person score {}'.format(track.id), 'score', "Person Score Track {}".format(track.id), frame, score.cpu().numpy()[0])
            scores = torch.cat(scores)
            pos = torch.cat(pos)
        else:
            pos = self.get_pos()
            boxes, scores = self.obj_detect.predict_boxes(pos)
            pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

        s = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
            # t.prev_pos = t.pos
            t.pos = pos[i].view(1, -1)

        scores_of_active_tracks = torch.Tensor(s[::-1]).to(device)

        return scores_of_active_tracks


    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).to(device)
        return pos


    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).to(device)
        return features


    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features


    def reid(self, blob, new_det_pos, new_det_scores):
        """Tries to ReID inactive tracks with provided detections."""
        zeros = torch.zeros(0).to(device)

        new_det_features = [zeros for _ in range(len(new_det_pos))]

        if self.do_reid:
            new_det_features = self.reid_network.test_rois(
                blob['img'], new_det_pos).data

            if len(self.inactive_tracks) >= 1:
                # calculate appearance distances
                dist_mat, pos = [], []
                for t in self.inactive_tracks:
                    dist_mat.append(torch.cat([t.test_features(feat.view(1, -1)) for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]

                # calculate IoU distances
                iou = bbox_overlaps(pos, new_det_pos)
                iou_mask = torch.ge(iou, self.reid_iou_threshold)
                iou_neg_mask = ~iou_mask
                # make all impossible assignments to the same add big value
                dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
                dist_mat = dist_mat.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        print(f"Reidying track {t.id}")
                        t.count_inactive = 0
                        t.pos = new_det_pos[c].view(1, -1)
                        t.reset_last_pos()
                        t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().to(device)
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).to(device)
                    new_det_scores = torch.zeros(0).to(device)
                    new_det_features = torch.zeros(0).to(device)

        return new_det_pos, new_det_scores, new_det_features


    def get_appearances(self, blob):
        """Uses the siamese CNN to get the features for all active tracks."""
        new_features = self.reid_network.test_rois(blob['img'], self.get_pos()).data
        return new_features


    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))


    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations, self.termination_eps)
            cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)#, inputMask=None,
                                                   #gaussFiltSize=1)
            warp_matrix = torch.from_numpy(warp_matrix)

            for t in self.tracks:
                t.pos = warp_pos(t.pos, warp_matrix)
            # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

            if self.do_reid:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            if self.motion_model_cfg['enabled']:
                for t in self.tracks:
                    for i in range(len(t.last_pos)):
                        t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)


    def motion_step(self, track):
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
        else:
            track.pos = track.pos + track.last_v


    def motion(self):
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for t in self.tracks:
            last_pos = list(t.last_pos)

            # avg velocity between each pair of consecutive positions in t.last_pos
            if self.motion_model_cfg['center_only']:
                vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
            else:
                vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

            t.last_v = torch.stack(vs).mean(dim=0)
            self.motion_step(t)

        if self.do_reid:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    self.motion_step(t)


    def step(self, blob, frame=1):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.clone())

        ###########################
        # Look for new detections #
        ###########################

        self.obj_detect.load_image(blob['img'])
        if self.public_detections:
            dets = blob['dets'].squeeze(dim=0)
            if dets.nelement() > 0:
                boxes, scores = self.obj_detect.predict_boxes(dets)
            else:
                boxes = scores = torch.zeros(0).cuda()
        else:
            boxes, scores = self.obj_detect.detect(blob['img'])

        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

            # Filter out tracks that have too low person score
            inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
        else:
            inds = torch.zeros(0).cuda()

        # Are there any bounding boxes that have a high enough person (class 1) classification score.
        if inds.nelement() > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()

        ##################
        # Predict tracks #
        ##################

        if len(self.tracks):
            # align
            if self.do_align:
                self.align(blob)

            # apply motion model
            if self.motion_model_cfg['enabled']:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]

            # regress
            person_scores = self.regress_tracks(blob, frame=frame)

            if len(self.tracks):
                # create nms input

                # nms here if tracks overlap
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

                for i, track in enumerate(self.tracks):
                    if i in keep:
                        track.frames_since_active += 1
                        if self.finetuning_config["finetune_repeatedly"]:
                            if np.mod(track.frames_since_active, self.finetuning_config["finetuning_interval"]) == 0:
                                box_head_copy = self.get_box_head()
                                box_predictor_copy = self.get_box_predictor()
                                other_pedestrians_bboxes = torch.Tensor([]).to(device)
                                for j in range(len(self.tracks)):
                                    if j != i:
                                        other_pedestrians_bboxes = torch.cat((other_pedestrians_bboxes, self.tracks[j].pos))
                                track.finetune_detector(
                                    self.obj_detect.roi_heads.box_roi_pool,
                                    self.obj_detect.fpn_features,
                                    self.finetuning_config,
                                    box_head_copy,
                                    box_predictor_copy,
                                    additional_dets=other_pedestrians_bboxes
                                )

                if keep.nelement() > 0:
                    if self.do_reid:
                        new_features = self.get_appearances(blob)
                        self.add_features(new_features)
        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            keep = nms(det_pos, det_scores, self.detection_nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            # check with every track in a single run (problem if tracks delete each other)
            for t in self.tracks:
                nms_track_pos = torch.cat([t.pos, det_pos])
                nms_track_scores = torch.cat(
                    [torch.tensor([2.0]).to(det_scores.device), det_scores])
                keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)

                keep = keep[torch.ge(keep, 1)] - 1

                det_pos = det_pos[keep]
                det_scores = det_scores[keep]
                if keep.nelement() == 0:
                    break

        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # try to reidentify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features, blob['img'][0])

        ####################
        # Generate Results #
        ####################

        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score])])

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]


    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps, im_info,
                 transformed_image_size):
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
        self.box_predictor = None
        self.box_head = None
        self.scale = self.im_info[0] / self.transformed_image_size[0][0]
        self.plotter = None
        self.checkpoints = dict()

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
    def generate_training_set_regession(self, gt_pos, max_displacement, batch_size=8, plot=False, plot_args=None):
        gt_pos = gt_pos.to(device)
        random_displaced_bboxes = replicate_and_randomize_boxes(gt_pos,
                                                                batch_size=batch_size,
                                                                max_displacement=max_displacement).to(device)

        training_boxes = clip_boxes(random_displaced_bboxes, self.im_info)

        if plot and plot_args:
            plot_bounding_boxes(self.im_info, gt_pos, plot_args[0], training_boxes.numpy(), plot_args[1], plot_args[2])

        return training_boxes

    def generate_training_set_classification(self, batch_size, additional_dets, shuffle=False):
        num_positive_examples = int(batch_size / 2)
        positive_examples = self.generate_training_set_regession(self.pos,
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
                negative_example = self.generate_training_set_regession(additional_dets[i].view(1, -1),
                                                                        0.0,
                                                                        batch_size=num_negative_example).to(device)
                negative_example = clip_boxes(negative_example, self.im_info)
                # negative_example = additional_dets[i].view(1, -1).repeat(num_negative_example, 1)
                negative_example_and_label = torch.cat((negative_example, torch.zeros([num_negative_example, 1]).to(device)), dim=1)
                boxes = torch.cat((boxes, negative_example_and_label))
        if shuffle:
            boxes = boxes[torch.randperm(boxes.size(0))]
        return boxes

    def forward_pass(self, boxes, box_roi_pool, fpn_features, eval=False, scores=False):
        if eval:
            self.box_predictor.eval()
            self.box_head.eval()
        boxes_resized = resize_boxes(boxes[:, 0:4], self.im_info, self.transformed_image_size[0])
        proposals = [boxes_resized]
        roi_pool_feat = box_roi_pool(fpn_features, proposals, self.im_info)
        # Only train the box prediction head
        with torch.no_grad():
            feat = self.box_head(roi_pool_feat)
        class_logits, _ = self.box_predictor(feat)
        if scores:
            pred_scores = F.softmax(class_logits, -1)
            if eval:
                self.box_predictor.train()
                self.box_head.train()
            return pred_scores[:, 1:].squeeze(dim=1).detach()
        loss = F.cross_entropy(class_logits, boxes[:, 4].long())
        if eval:
            self.box_predictor.train()
            self.box_head.train()
        return loss

    def finetune_detector(self, box_roi_pool, fpn_features,
                          finetuning_config, box_head, box_predictor, additional_dets=None, early_stopping=True):
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.box_predictor.train()
        self.box_head.train()
        optimizer = torch.optim.Adam(list(self.box_predictor.parameters()),
                                     lr=float(finetuning_config["learning_rate"]))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=1)


        if finetuning_config["validate"]:
            if not self.plotter:
                self.plotter = VisdomLinePlotter(env_name='training')
            additional_dets = additional_dets[:-1]
            val_dets = additional_dets[-1:]
            validation_boxes = self.generate_training_set_classification(float(int(finetuning_config["batch_size_val"])),
                                                                         val_dets).to(device)
        print("Finetuning track {}".format(self.id))
        for i in range(int(finetuning_config["iterations"])):

            optimizer.zero_grad()

            training_boxes = self.generate_training_set_classification(int(finetuning_config["batch_size"]),
                                                                       additional_dets)

            loss = self.forward_pass(training_boxes, box_roi_pool, fpn_features, eval=False)
            #print('Finished iteration {} --- Loss {}'.format(i, loss.item()))

            if np.mod(i, int(finetuning_config["iterations_per_validation"])) == 0 and finetuning_config["validate"]:
                val_loss = self.forward_pass(validation_boxes, box_roi_pool, fpn_features, eval=True)
                self.plotter.plot('loss', 'val', "Bbox Loss Track {}".format(self.id), i, val_loss.item())

            if early_stopping:
                scores = self.forward_pass(training_boxes, box_roi_pool, fpn_features, scores=True, eval=True)

                if finetuning_config["validate"]:
                    print(i)
                    print('Average score of positive examples: {}'.format(torch.mean(scores[:16])))
                    print('Average score of negative examples: {}\n'.format(torch.mean(scores[16:])))
                    self.plotter.plot('loss', 'positive', 'Class Loss Evaluation Track {}'.format(self.id), i, torch.mean(scores[:16]))
                    for sample in range(16, 32):
                        self.plotter.plot('loss', 'negative {}'.format(sample), 'Class Loss Evaluation Track {}'.format(self.id), i, scores[sample])

                if scores[0] - torch.max(scores[16:]) > 0.1 and scores[0] > 0.8:
                    print('Stopping because difference between positive score and maximum negative score is {}'.format(scores[0] - torch.max(scores[16:])))
                    break



            loss.backward()
            optimizer.step()
            scheduler.step()

        self.box_predictor.eval()
        self.box_head.eval()

        dets = torch.cat((self.pos, additional_dets))
        print(self.forward_pass(dets, box_roi_pool, fpn_features, scores=True))

#
#                    idf1       idp       idr    recall  precision  num_unique_objects  mostly_tracked  partially_tracked  mostly_lost  num_false_positives  num_misses  num_switches  num_fragmentations      mota      motp
#MOT17-02-FRCNN  0.458597  0.784569  0.323987  0.411980   0.997654                  62               8                 32           22                   18       10926            57                  65  0.407944  0.079305
#MOT17-04-FRCNN  0.712063  0.904892  0.586980  0.647265   0.997828                  83              32                 29           22                   67       16775            21                  28  0.645415  0.095695
#MOT17-05-FRCNN  0.633866  0.859832  0.501952  0.573804   0.982912                 133              32                 65           36                   69        2948            38                  60  0.558335  0.142563
#MOT17-09-FRCNN  0.536235  0.681831  0.441878  0.641878   0.990438                  26              11                 13            2                   33        1907            23                  31  0.631362  0.086603
#MOT17-10-FRCNN  0.653085  0.768088  0.568035  0.723810   0.978726                  57              28                 26            3                  202        3546            66                 119  0.702936  0.145639
#MOT17-11-FRCNN  0.632742  0.770061  0.536986  0.690229   0.989818                  75              24                 33           18                   67        2923            26                  25  0.680373  0.081523
#MOT17-13-FRCNN  0.726847  0.840207  0.640440  0.741797   0.973180                 110              59                 40           11                  238        3006            59                  84  0.716286  0.130974
#OVERALL         0.650191  0.839572  0.530522  0.625716   0.990220                 546             194                238          114                  694       42031           290                 412  0.616953  0.105742
#
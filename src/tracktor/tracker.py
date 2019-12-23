from collections import deque, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import cv2
from torchvision.ops import box_iou

from tracktor.training_set_generation import replicate_and_randomize_boxes
from tracktor.visualization import plot_compare_bounding_boxes, VisdomLinePlotter, plot_bounding_boxes, \
    parse_ground_truth
from tracktor.utils import clip_boxes
from tracktor.utils import bbox_overlaps, warp_pos, get_center, get_height, get_width, make_pos

from torchvision.ops.boxes import clip_boxes_to_image, nms, box_iou
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.transform import resize_boxes

import matplotlib
import matplotlib.pyplot as plt
if not torch.cuda.is_available():
    matplotlib.use('TkAgg')

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
        for i in range(num_new):
            track = Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i,
                          new_det_features[i].view(1, -1), self.inactive_patience, self.max_features_num,
                          self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1,
                          image.size()[1:3], self.obj_detect.image_size)

            if self.finetuning_config["enabled"]:
                box_head_copy = TwoMLPHead(self.obj_detect.backbone.out_channels *
                                           self.obj_detect.roi_heads.box_roi_pool.output_size[0] ** 2,
                                           representation_size=1024).to(device)
                box_predictor_copy = FastRCNNPredictor(1024, 2).to(device)

                box_head_copy.load_state_dict(self.bbox_head_weights)
                box_predictor_copy.load_state_dict(self.bbox_predictor_weights)

                track.finetune_detector(self.obj_detect.roi_heads.box_roi_pool,
                                        self.obj_detect.fpn_features,
                                        new_det_pos[i],
                                        self.obj_detect.roi_heads.box_coder.decode,
                                        image,
                                        self.finetuning_config,
                                        box_head=box_head_copy,
                                        box_predictor=box_predictor_copy,
                                        plot=False)
            self.tracks.append(track)

        self.track_num += num_new

    #@staticmethod
    #def compare_weights(m1, m2):
    #    for p1, p2 in zip(m1.parameters(), m2.parameters()):
    #        if p1.data.ne(p2.data).sum() > 0:
     #           return False
     #   return True
    @staticmethod
    def compare_models(m1, m2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(m1.state_dict().items(), m2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                continue
            else:
                return False
        return True

    def regress_tracks(self, blob, plot_compare=False, frame=None):
        """Regress the position of the tracks and also checks their scores."""
        if self.finetuning_config["enabled"]:
            scores = []
            pos = []
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
            scores = torch.cat(scores)
            pos = torch.cat(pos)
        else:
            pos = self.get_pos()
            boxes, scores = self.obj_detect.predict_boxes(pos)
            pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

        if frame==526 or frame == 527 or frame == 528:
            print([track.id for track in self.tracks])
            print(scores)
            if frame == 528:
                input("HI")

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
            pos = torch.zeros(0).cuda()
        return pos


    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
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

        num_tracks = 0
        nms_inp_reg = torch.zeros(0).to(device)

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
                if frame==2:
                    print(box_iou(*[t.pos() for t in self.tracks if t.id in [2, 0]]))
                # nms here if tracks overlap
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)
                if frame == 526 or frame == 527 or frame == 528:
                    print(keep)
                    print(box_iou(*[t.pos() for t in self.tracks if t.id in [15, 25]]))
                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

                if frame == 526 or frame == 527 or frame == 528:
                    print([track.id for track in self.tracks])

                for i, track in enumerate(self.tracks):
                    if i in keep:
                        track.frames_since_active += 1
                        if self.finetuning_config["finetune_repeatedly"]:
                            if np.mod(track.frames_since_active, self.finetuning_config["finetuning_interval"]) == 0:
                                track.finetune_detector(
                                    self.obj_detect.roi_heads.box_roi_pool,
                                    self.obj_detect.fpn_features,
                                    track.pos.squeeze(0),
                                    self.obj_detect.roi_heads.box_coder.decode,
                                    blob['img'][0],
                                    self.finetuning_config
                                )
                        if self.finetuning_config["validation_over_time"]:
                            if np.mod(track.frames_since_active, self.finetuning_config["validation_interval"]) == 0:
                                for checkpoint, models in track.checkpoints.items():
                                    test_rois = track.generate_training_set(self.finetuning_config["max_displacement"], batch_size=128)
                                    box_pred_val, _ = self.obj_detect.predict_boxes(test_rois[:, 0:4],
                                                                                                  box_head=models[0],
                                                                                                  box_predictor=models[1])
                                    # plot_bounding_boxes(blob['img'][0].size()[1:3],
                                    #     track.pos,
                                    #     blob['img'][0],
                                    #     box_pred_val,
                                    #     frame,
                                    #     track.id,
                                    #     validate=True)

                                    annotated_boxes = parse_ground_truth(frame).type(torch.FloatTensor)
                                    index_likely_bounding_box = np.argmax(box_iou(track.pos, annotated_boxes))

                                    annotated_likely_ground_truth_bounding_box = annotated_boxes[index_likely_bounding_box, :]

                                    criterion_regressor = torch.nn.SmoothL1Loss()

                                    loss = criterion_regressor(box_pred_val,
                                                     annotated_likely_ground_truth_bounding_box.repeat(128, 1))
                                    track.plotter.plot('loss', 'val {}'.format(checkpoint), 'Class Loss track {}'.format(i),
                                                       track.frames_since_active, loss.item())

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

    def generate_training_set(self, max_displacement, batch_size=8, plot=False, plot_args=None):
        gt_pos = self.pos.to(device)
        # displacement should be the realistic displacement of the bounding box from t-frame to the t+1-frame.
        # TODO implement check that the randomly generated box has largest IoU with gt_pos compared to all other
        # detections.
        random_displaced_bboxes = replicate_and_randomize_boxes(gt_pos,
                                                                batch_size=batch_size,
                                                                max_displacement=max_displacement).to(device)

        training_boxes = clip_boxes(random_displaced_bboxes, self.im_info)

        if plot and plot_args:
            plot_bounding_boxes(self.im_info, gt_pos, plot_args[0], training_boxes.numpy(), plot_args[1], plot_args[2])

        return training_boxes

    def finetune_detector(self, box_roi_pool, fpn_features, gt_box, bbox_pred_decoder, image,
                          finetuning_config, box_head=None, box_predictor=None, plot=False):
        if box_head is not None:
            self.box_head = box_head

        if box_predictor is not None:
            self.box_predictor = box_predictor

        self.box_predictor.train()
        self.box_head.train()
        optimizer = torch.optim.Adam(list(self.box_predictor.parameters()),
                                     lr=float(finetuning_config["learning_rate"]) if box_head is not None
                                     else float(finetuning_config["finetuning_interval_learning_rate"]))
        criterion = torch.nn.SmoothL1Loss()

        if isinstance(fpn_features, torch.Tensor):
            fpn_features = OrderedDict([(0, fpn_features)])

        if finetuning_config["validation_over_time"]:
            if not self.plotter:
                self.plotter = VisdomLinePlotter()
            validation_boxes = self.generate_training_set(float(finetuning_config["max_displacement"]),
                                                          batch_size=int(finetuning_config["batch_size_val"]),
                                                          plot=plot,
                                                          plot_args=(image, "val", self.id)).to(device)
            validation_boxes_resized = resize_boxes(
                validation_boxes, self.im_info, self.transformed_image_size[0])
            proposals_val = [validation_boxes_resized]
            roi_pool_feat_val = box_roi_pool(fpn_features, proposals_val, self.im_info)
            plotter = VisdomLinePlotter()

        self.checkpoints[0] = [box_head, box_predictor]

        for i in range(int(finetuning_config["iterations"])):

            if finetuning_config["validation_over_time"]:
                if np.mod(i+1, finetuning_config["checkpoint_interval"]) == 0:
                    self.checkpoints[i+1] = [box_head, box_predictor]

            optimizer.zero_grad()
            training_boxes = self.generate_training_set(float(finetuning_config["max_displacement"]),
                                                        batch_size=int(finetuning_config["batch_size"]),
                                                        plot=plot,
                                                        plot_args=(image, i, self.id)).to(device)

            boxes = resize_boxes(
                training_boxes, self.im_info, self.transformed_image_size[0])
            scaled_gt_box = resize_boxes(
                gt_box.unsqueeze(0), self.im_info, self.transformed_image_size[0]).squeeze(0)
            proposals = [boxes]

            with torch.no_grad():
                roi_pool_feat = box_roi_pool(fpn_features, proposals, self.im_info)
                # feed pooled features to top model
                pooled_feat = self.box_head(roi_pool_feat)

            # compute bbox offset
            _, bbox_pred = self.box_predictor(pooled_feat)

            pred_boxes = bbox_pred_decoder(bbox_pred, proposals)
            pred_boxes = pred_boxes[:, 1:].squeeze(dim=1)

            if np.mod(i, int(finetuning_config["iterations_per_validation"])) == 0 and finetuning_config["validate"]:
                pooled_feat_val = self.box_head(roi_pool_feat_val)
                _, bbox_pred_val = self.box_predictor(pooled_feat_val)
                pred_boxes_val = bbox_pred_decoder(bbox_pred_val, proposals_val)
                pred_boxes_val = pred_boxes_val[:, 1:].squeeze(dim=1)
                #plot_bounding_boxes(self.im_info,
                #                    gt_box.unsqueeze(0),
                #                    image,
                #                    resize_boxes(pred_boxes_val, self.transformed_image_size[0], self.im_info),
                #                    i,
                #                    self.id,
                #                    validate=True)
                val_loss = criterion(pred_boxes_val, scaled_gt_box.repeat(int(finetuning_config["batch_size_val"]), 1))
                plotter.plot('loss', 'val', "Bbox Loss Track {}".format(self.id), i, val_loss.item())

            loss = criterion(pred_boxes, scaled_gt_box.repeat(int(finetuning_config["batch_size"]), 1))
            loss.backward()
            optimizer.step()
            print('Finished iteration {} --- Loss {}'.format(i, loss.item()))

        self.box_predictor.eval()
        self.box_head.eval()

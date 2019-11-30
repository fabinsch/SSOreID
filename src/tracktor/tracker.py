from collections import deque, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import cv2

from tracktor.training_set_generation import replicate_and_randomize_boxes
from .utils import bbox_overlaps, bbox_transform_inv, clip_boxes
from helper.csrc.wrapper.nms import nms
from .utils import bbox_overlaps, warp_pos, get_center, get_height, get_width, make_pos

from torchvision.ops.boxes import clip_boxes_to_image, nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

import matplotlib

if not torch.cuda.is_available():
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        self.finetune_appearance_models = tracker_cfg['finetune_appearance_models']
        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']

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

            if self.finetune_appearance_models:
                track.generate_training_set(image, plot=False)
                box_head_copy = TwoMLPHead(self.obj_detect.backbone.out_channels *
                                           self.obj_detect.roi_heads.box_roi_pool.output_size[0] ** 2,
                                           representation_size=1024)
                box_predictor_copy = FastRCNNPredictor(1024, 2)

                box_head_copy.load_state_dict(self.obj_detect.roi_heads.box_head.state_dict())
                box_predictor_copy.load_state_dict(self.obj_detect.roi_heads.box_predictor.state_dict())
                if torch.cuda.is_available():
                    box_head_copy = box_head_copy.cuda()
                    box_predictor_copy = box_predictor_copy.cuda()
                track.finetune_detector(box_head_copy, self.obj_detect.roi_heads.box_roi_pool,
                                        box_predictor_copy, self.obj_detect.fpn_features, new_det_pos[i],
                                        self.obj_detect.roi_heads.box_coder.decode)
            self.tracks.append(track)

        self.track_num += num_new


    def regress_tracks(self, blob):
        """Regress the position of the tracks and also checks their scores."""
        if self.finetune_appearance_models:
            scores = []
            pos = []
            for i, track in enumerate(self.tracks):
                # Regress with finetuned bbox head for each track
                box, score = self.obj_detect.predict_boxes(track.pos,
                                                           box_head=track.box_head,
                                                           box_predictor=track.box_predictor)
                scores.append(score)
                bbox = clip_boxes_to_image(box, blob['img'].shape[-2:])
                pos.append(bbox)
            scores = torch.cat(scores)
            pos = torch.cat(pos)

        else:
            pos = self.get_pos()
            boxes, scores = self.obj_detect.predict_boxes(pos)
            pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])
        print(scores)
        print(pos)
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

        scores_of_active_tracks = torch.Tensor(s[::-1])
        if torch.cuda.is_available():
            scores_of_active_tracks.cuda()
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
        zeros = torch.zeros(0)
        if torch.cuda.is_available():
            zeros.cuda()
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

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long()
                if torch.cuda.is_available():
                    keep.cuda()
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0)
                    new_det_scores = torch.zeros(0)
                    new_det_features = torch.zeros(0)
                    if torch.cuda.is_available():
                        new_det_pos = new_det_pos.cuda()
                        new_det_scores = new_det_scores.cuda()
                        new_det_features = new_det_features.cuda()

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
            cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria, inputMask=None,
                                                   gaussFiltSize=1)
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


    def step(self, blob):
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
        nms_inp_reg = torch.zeros(0)
        if torch.cuda.is_available():
            nms_inp_reg.cuda()

        if len(self.tracks):
            # align
            if self.do_align:
                self.align(blob)

            # apply motion model
            if self.motion_model_cfg['enabled']:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]

            # regress
            person_scores = self.regress_tracks(blob)

            if len(self.tracks):
                # create nms input

                # nms here if tracks overlap
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

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
        self.last_v = torch.Tensor([])
        self.transformed_image_size = transformed_image_size
        self.gt_id = None
        self.im_info = im_info
        self.box_predictor = None
        self.box_head = None

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

    def generate_training_set(self, image, batch_size=8, plot=False):
        gt_pos = self.pos
        random_displaced_bboxes = replicate_and_randomize_boxes(gt_pos, batch_size=8)

        if torch.cuda.is_available():
            gt_pos = gt_pos.cuda()
            random_displaced_bboxes = random_displaced_bboxes.cuda()
        self.training_boxes = clip_boxes(random_displaced_bboxes, self.im_info)

        if plot:
            rectangles = self.training_boxes.numpy()
            num_rectangles = len(rectangles)
            h, w = self.im_info
            image = image.squeeze()
            image = image.permute(1,2,0)
            fig, ax = plt.subplots(1)
            ax.imshow(image, cmap='gist_gray_r')
            gt_pos_np = gt_pos.numpy()
            ax.add_patch(
                        plt.Rectangle((gt_pos_np[0, 0], gt_pos_np[0, 1]),
                                      gt_pos_np[0, 2] - gt_pos_np[0, 0],
                                      gt_pos_np[0, 3] - gt_pos_np[0, 1], fill=False,
                            linewidth=1.3, color='blue')
                    )
            for i in range(num_rectangles):
                ax.add_patch(
                    plt.Rectangle((rectangles[i, 0], rectangles[i, 1]),
                                  rectangles[i, 2] - rectangles[i, 0],
                                  rectangles[i, 3] - rectangles[i, 1], fill=False,
                                  linewidth=1.3, color='blue')
                )

            plt.show()

    def finetune_detector(self, box_head, box_roi_pool, box_predictor, fpn_features, gt_box, bbox_pred_decoder, epochs=10):


        optimizer = torch.optim.Adam(list(box_predictor.parameters()) + list(box_head.parameters()), lr=0.0001)
        criterion = torch.nn.SmoothL1Loss()

        if isinstance(fpn_features, torch.Tensor):
            fpn_features = OrderedDict([(0, fpn_features)])

        # proposals, proposal_losses = self.rpn(images, features, targets)
        from torchvision.models.detection.transform import resize_boxes
        boxes = resize_boxes(
            self.training_boxes, self.im_info, self.transformed_image_size[0])
        proposals = [boxes]

        with torch.no_grad():
            roi_pool_feat = box_roi_pool(fpn_features, proposals, self.im_info)

        for i in range(epochs):

            # feed pooled features to top model
            pooled_feat = box_head(roi_pool_feat)
            pooled_feat = pooled_feat.squeeze()

            # compute bbox offset
            _, bbox_pred = box_predictor(pooled_feat)

            pred_boxes = bbox_pred_decoder(bbox_pred, proposals)


            optimizer.zero_grad()
            loss = criterion(pred_boxes, gt_box)
            loss.backward()
            optimizer.step()
            print('Finished epoch {} --- Loss {}'.format(i, loss.item()))

        # TODO plot bounding boxes after training

        self.box_predictor = box_predictor
        self.box_head = box_head

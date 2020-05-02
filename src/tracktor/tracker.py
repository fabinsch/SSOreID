import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import cv2
from collections import defaultdict
import datetime

from tracktor.track import Track
from tracktor.visualization import plot_compare_bounding_boxes, VisdomLinePlotter, plot_bounding_boxes
from tracktor.utils import bbox_overlaps, warp_pos, get_center, get_height, get_width, make_pos, EarlyStopping
from tracktor.live_dataset import InactiveDataset, IndividualDataset

from torchvision.ops.boxes import clip_boxes_to_image, nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torch.nn import functional as F

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
        self.reid_siamese = tracker_cfg['reid_siamese']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']
        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']
        self.finetuning_config = tracker_cfg['finetuning']
        if self.finetuning_config["for_tracking"] or self.finetuning_config["for_reid"]:
            self.bbox_predictor_weights = self.obj_detect.roi_heads.box_predictor.state_dict()
            self.bbox_head_weights = self.obj_detect.roi_heads.box_head.state_dict()

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        self.inactive_tracks_temp = []
        self.inactive_number_changes = 0
        self.box_head_classification = None
        self.box_predictor_classification = None
        self.training_set = None
        now = datetime.datetime.now()
        self.run_name = now.strftime("%Y-%m-%d %H:%M")
        self.num_reids = 0
        self.checkpoints = {}
        self.killed_this_step = []
        self.num_training = 0
        self.train_on = []
        self.count_killed_this_step_reid = 0
        self.c_just_one_frame_active = 0

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        self.inactive_tracks_temp = []
        self.inactive_number_changes = 0
        self.num_reids = 0
        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            if t.frames_since_active > 1:
                t.pos = t.last_pos[-1]
                self.inactive_number_changes += 1
                self.killed_this_step.append(t.id)
            else:
                self.c_just_one_frame_active += 1
                #tracks.remove(t)
                t.pos = t.last_pos[-1]
                self.inactive_number_changes += 1
                self.killed_this_step.append(t.id)
        self.inactive_tracks += tracks


    def add(self, new_det_pos, new_det_scores, image, frame, new_det_features=None):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        old_tracks = self.get_pos()
        box_roi_pool = self.obj_detect.roi_heads.box_roi_pool
        for i in range(num_new):
            track = Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i,
                          new_det_features[i].view(1, -1) if new_det_features else None, self.inactive_patience, self.max_features_num,
                          self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1,
                          image.size()[1:3], self.obj_detect.image_size, self.finetuning_config["batch_size"],
                          box_roi_pool=box_roi_pool, keep_frames=self.finetuning_config['keep_frames'])

            other_pedestrians_bboxes = torch.cat((new_det_pos[:i], new_det_pos[i + 1:], old_tracks))
            track.update_training_set_classification(self.finetuning_config['batch_size'],
                                                 other_pedestrians_bboxes,
                                                 self.obj_detect.fpn_features,
                                                 include_previous_frames=True)

            self.tracks.append(track)

        self.track_num += num_new

    def get_box_predictor_(self, n=2):
        """Get a box predictor with number of output neurons corresponding to number of inactive tracks + 1 for others"""
        box_predictor = FastRCNNPredictor(1024, n+1).to(device)
        #box_predictor.load_state_dict(self.bbox_predictor_weights)
        return box_predictor

    def get_box_predictor(self):
        box_predictor = FastRCNNPredictor(1024, 2).to(device)
        box_predictor.load_state_dict(self.bbox_predictor_weights)
        return box_predictor

    def get_box_head(self, reset=True):
        if reset:
            box_head =  TwoMLPHead(self.obj_detect.backbone.out_channels *
                                       self.obj_detect.roi_heads.box_roi_pool.output_size[0] ** 2,
                                       representation_size=1024).to(device)
            box_head.load_state_dict(self.bbox_head_weights)
        else:
            box_head = self.box_head_classification  # do not start again with pretrained weights
        return box_head

    def regress_tracks(self, blob, plot_compare=False, frame=None):
        """Regress the position of the tracks and also checks their scores."""
        pos = self.get_pos()

        # regress
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
                t.pos = pos[i].view(1, -1)  # here in original implementation
            # t.prev_pos = t.pos
            t.pos = pos[i].view(1, -1)  # here adrian and caro


        scores_of_active_tracks = torch.Tensor(s[::-1]).to(device)

        return scores_of_active_tracks


    def get_pos(self, active=True):
        """Get the positions of all active/inactive tracks."""
        if active:
            tracks = self.tracks
        else:
            tracks = self.inactive_tracks
        if len(tracks) == 1:
            pos = tracks[0].pos
        elif len(tracks) > 1:
            pos = torch.cat([t.pos for t in tracks], 0)
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

    def reid_by_finetuned_model(self, new_det_pos, new_det_scores, frame):
        # IDEA: evaluate all inactive track models on the new detections
        # reidentify a track, when the model has a significantly higher score on this new detection than on other detections
        active_tracks = self.get_pos()
        if len(new_det_pos.size()) > 1 and len(self.inactive_tracks) > 0:
            remove_inactive = []
            det_index_to_candidate = defaultdict(list)
            assigned = []
            score_matrix = torch.tensor([]).to(device) # 1 row: scores for a new detection by the current inactive tracks
            #idea: go over the detections, check the scores of the classifiers wheter one is significantly higher
            for inactive_track in self.inactive_tracks:
                boxes, scores = self.obj_detect.predict_boxes(new_det_pos,
                                                             box_predictor_classification=inactive_track.box_predictor_classification,
                                                             box_head_classification=inactive_track.box_head_classification)

                if score_matrix.size()[0] == 0:
                    score_matrix = scores.unsqueeze(1)
                else:
                    if len(scores.size()) == 1:
                        scores = scores.unsqueeze(1)
                    score_matrix = torch.cat([score_matrix, scores], dim=1)
            print(f'Score matrix: {score_matrix.data.cpu().numpy()}')

            for track_index in range(len(self.inactive_tracks)):

                track_scores = score_matrix[:, track_index]
                highest_score_index = torch.argmax(track_scores)
                highest_score = torch.max(track_scores)
                track_scores[highest_score_index] = 0
                second_highest_score = torch.max(track_scores)
                distance_to_second_highest_score = highest_score - second_highest_score
                if distance_to_second_highest_score > 0.2 and highest_score > 0.95:
                    inactive_track = self.inactive_tracks[track_index]
                    det_index_to_candidate[int(highest_score_index.cpu().numpy())].append((inactive_track, highest_score))

            for det_index, candidates in det_index_to_candidate.items():
                if len(candidates) == 1:
                    candidate = candidates[0]
                    inactive_track = candidate[0]
                    self.tracks.append(inactive_track)
                    print(f"Reidying track {inactive_track.id} in frame {frame} with score {candidate[1]}")
                    inactive_track.count_inactive = 0
                    inactive_track.pos = new_det_pos[det_index].view(1, -1)
                    inactive_track.reset_last_pos()
                    assigned.append(det_index)
                    remove_inactive.append(inactive_track)

            for inactive_track in remove_inactive:
                self.inactive_number_changes += 1
                self.inactive_tracks.remove(inactive_track)
                inactive_track.update_training_set_classification(self.finetuning_config['batch_size'],
                                                                  active_tracks,
                                                                  self.obj_detect.fpn_features,
                                                                  include_previous_frames=True)

            keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().to(device)
            if keep.nelement() > 0:
                new_det_pos = new_det_pos[keep]
                new_det_scores = new_det_scores[keep]
            else:
                new_det_pos = torch.zeros(0).to(device)
                new_det_scores = torch.zeros(0).to(device)

        return new_det_pos, new_det_scores


    def reid_by_finetuned_model_(self, new_det_pos, new_det_scores, frame):
        """Do reid with one model predicting the score for each inactive track
        Note: work with self.inactive_tracks_temp because model was trained on those, self.inactive_tracks might
        already have been changed by regress_tracks method
        """
        assert self.inactive_tracks_temp==self.inactive_tracks
        active_tracks = self.get_pos()
        if len(new_det_pos.size()) > 1 and len(self.inactive_tracks_temp) > 0:
            remove_inactive = []
            det_index_to_candidate = defaultdict(list)
            inactive_to_det = defaultdict(list)
            assigned = []
            inactive_tracks = self.get_pos(active=False)

            boxes, scores = self.obj_detect.predict_boxes(new_det_pos,
                                                          box_predictor_classification=self.box_predictor_classification,
                                                          box_head_classification=self.box_head_classification,
                                                          pred_multiclass=True)

            if frame==420:
                print('\n scores for REID: {}'.format(scores))

            # calculate IoU distances
            iou = bbox_overlaps(new_det_pos, inactive_tracks)
            # iou has just values for the inactive tracks -> extend for others class
            iou = torch.cat((torch.ones(iou.shape[0],1).to(device), iou), dim=1)
            iou_mask = torch.ge(iou, self.reid_iou_threshold)
            scores = scores * iou_mask
            scores = scores.cpu().numpy()
            max = scores.max(axis=1)
            max_idx = scores.argmax(axis=1)
            scores[:, max_idx] = 0
            max2 = scores.max(axis=1)
            #max_idx2 = scores.argmax(axis=1)
            dist = max - max2

            if frame==4200:
                # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
                self.inactive_tracks[max_idx[0]].add_classifier(self.box_predictor_classification, self.box_head_classification)
                #print('d')

            for i, d in enumerate(dist):
                if max[i] > self.finetuning_config['reid_score_threshold']:
                    # idx = 0 means unknown background people, idx=1,2,.. is inactive
                    if max_idx[i] == 0:
                        print('\n no reid because class 0 has score {}'.format(max[i]))

                    else:
                        inactive_track = self.inactive_tracks_temp[max_idx[i]-1]
                        det_index_to_candidate[i].append((inactive_track, max[i]))
                        inactive_to_det[max_idx[i]-1].append(i)

                elif max[i] > 0.0 :
                    print('\n no reid with score {}'.format(max[i]))

            for det_index, candidates in det_index_to_candidate.items():
                candidate = candidates[0]
                inactive_track = candidate[0]
                # get the position of the inactive track in inactive_tracks
                # if just one track, position "is 1" because 0 is unknown background person
                # important for check in next if statement
                inactive_id_in_list = self.inactive_tracks_temp.index(inactive_track)

                if len(inactive_to_det[inactive_id_in_list]) == 1:
                    # make sure just 1 new detection per inactive track
                    self.tracks.append(inactive_track)
                    print(f"\nReidying track {inactive_track.id} in frame {frame} with score {candidate[1]}")
                    print(' - it was trained on inactive tracks {}'.format([t.id for t in self.inactive_tracks_temp]))
                    self.num_reids += 1

                    if inactive_track.id in self.killed_this_step:
                        self.count_killed_this_step_reid += 1
                        print('\n track {} was killed and reid in frame {}'.format(inactive_track.id, self.im_index))

                    # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
                    if frame==4200:
                        inactive_track.add_classifier(self.box_predictor_classification, self.box_head_classification)

                    # reset inactive track
                    inactive_track.count_inactive = 1
                    inactive_track.pos = new_det_pos[det_index].view(1, -1)
                    inactive_track.reset_last_pos()

                    if self.finetuning_config['reset_dataset']:
                        inactive_track.frames_since_active = 0
                        inactive_track.training_set = IndividualDataset(inactive_track.id, 64, 40)

                    assigned.append(det_index)
                    remove_inactive.append(inactive_track)
                else:
                    print('\nerror, {} new det for 1 inactive track ID {}'.format(len(inactive_to_det[inactive_id_in_list]), inactive_track.id))
                    print(' - it was trained on inactive tracks {}'.format([t.id for t in self.inactive_tracks_temp]))

            for inactive_track in remove_inactive:
                self.inactive_tracks.remove(inactive_track)
                inactive_track.update_training_set_classification(self.finetuning_config['batch_size'],
                                                                  active_tracks,
                                                                  self.obj_detect.fpn_features,
                                                                  include_previous_frames=True)

            keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().to(device)
            if keep.nelement() > 0:
                new_det_pos = new_det_pos[keep]
                new_det_scores = new_det_scores[keep]
            else:
                new_det_pos = torch.zeros(0).to(device)
                new_det_scores = torch.zeros(0).to(device)

        return new_det_pos, new_det_scores


    def reid(self, blob, new_det_pos, new_det_features, new_det_scores):
        """Tries to ReID inactive tracks with provided detections."""
        zeros = torch.zeros(0).to(device)

        new_det_features = [zeros for _ in range(len(new_det_pos))]

        if self.reid_siamese:
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
                iou_mask = torch.ge(iou, self.reid_iou_threshold)   # wird nicht reided wenn iou größer als der iou threshold ist "To minimize the risk of false reIDs, weonly consider pairs of deactivated and new bounding boxeswith a sufficiently large IoU
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
                    self.inactive_number_changes += 1

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().to(device)
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).to(device)
                    new_det_scores = torch.zeros(0).to(device)
                    new_det_features = torch.zeros(0).to(device)

        return new_det_pos, new_det_scores


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

            if self.reid_siamese:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            if self.finetuning_config['for_reid']:
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

        if self.reid_siamese:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    self.motion_step(t)

        if self.finetuning_config['for_reid']:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    self.motion_step(t)

    def step(self, blob, frame=1):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        self.killed_this_step = []  # which track became inactive this step
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

                # nms here if tracks overlap, delete tracks with lower score if IoU overpasses threshold
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)
                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

                # calculate IoU distances
                iou = bbox_overlaps(self.get_pos(), self.get_pos())
                for i, track in enumerate(self.tracks):
                    #if i in keep:  # TODO nicht sicher ob des hier stimmt nochmal zu checken, von adrian und caro
                    if torch.sum(iou[i] > self.finetuning_config['train_iou_threshold']) > 1:
                        #print('IuO track {} is big, do not use for training'.format(track.id))
                        continue
                    track.frames_since_active += 1

                    # debug
                    if hasattr(track, 'box_predictor_classification_debug'):
                        boxes_debug, scores_debug = self.obj_detect.predict_boxes(track.pos, track.box_predictor_classification_debug,track.box_head_classification_debug, pred_multiclass=True)
                        print('\n scores {}'.format(scores_debug))

                    other_pedestrians_bboxes = torch.Tensor([]).to(device)
                    for j in range(len(self.tracks)):
                        if j != i:
                            assert self.tracks[j].id not in self.killed_this_step
                            other_pedestrians_bboxes = torch.cat((other_pedestrians_bboxes, self.tracks[j].pos))

                    if self.finetuning_config["build_up_training_set"] and np.mod(track.frames_since_active,
                                                    self.finetuning_config["feature_collection_interval"]) == 0:
                        track.update_training_set_classification(self.finetuning_config['batch_size'],
                                        other_pedestrians_bboxes,
                                        self.obj_detect.fpn_features,
                                        include_previous_frames=True)

        # train REID model new if change in active tracks happened
        if (self.inactive_tracks != self.inactive_tracks_temp):
            if self.finetuning_config["for_reid"]:
                box_head_copy_for_classifier = self.get_box_head(reset=True)  # get head and load weights
                box_predictor_copy_for_classifier = self.get_box_predictor_(n=len(self.inactive_tracks))  # get predictor with corrsponding output number
                self.finetune_classification(self.finetuning_config, box_head_copy_for_classifier,
                                             box_predictor_copy_for_classifier,
                                             early_stopping=self.finetuning_config[
                                                 'early_stopping_classifier'],
                                             killed_this_step=self.killed_this_step)

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
            new_det_features=None
            if self.reid_siamese:
                new_det_features = self.reid_network.test_rois(blob['img'], new_det_pos).data
                new_det_pos, new_det_scores = self.reid(blob, new_det_pos, new_det_features, new_det_scores)
            elif self.finetuning_config["for_reid"]:
                new_det_pos, new_det_scores = self.reid_by_finetuned_model_(new_det_pos, new_det_scores, frame)
            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, blob['img'][0], frame, new_det_features)

        ####################
        # Generate Results #
        ####################

        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score.cpu()])])

        for t in self.inactive_tracks:
            t.count_inactive += 1

            # number of inactive tracks changes when inactive_patience is overpassed
            if t.count_inactive > self.inactive_patience:
                self.inactive_number_changes += 1

        # delete track from inactive_tracks when inactive_patience is overpassed
        # for t in self.inactive_tracks:
        #     if not t.has_positive_area() or t.count_inactive > self.inactive_patience:
        #         self.inactive_tracks.remove(t)
        #         self.killed_this_step = []

        # delete track from inactive_tracks when inactive_patience is overpassed - Original version
        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]



    def get_results(self):
        return self.results


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
            return pred_scores.detach()
            #return pred_scores[:, 1:].squeeze(dim=1).detach()
        loss = F.cross_entropy(class_logits, scores.long())
        if eval:
            self.box_predictor_classification.train()
            self.box_head_classification.train()
        return loss


    def finetune_classification(self, finetuning_config, box_head_classification,
                                box_predictor_classification, early_stopping, killed_this_step):

        #train_acc_criterion = False
        # do not train when no tracks
        if len(self.inactive_tracks) == 0:
            self.inactive_tracks_temp = self.inactive_tracks.copy()
            return

        for t in self.inactive_tracks:
               t.training_set.post_process()

        self.training_set = InactiveDataset(batch_size=finetuning_config['batch_size'], killed_this_step=killed_this_step)
        self.box_head_classification = box_head_classification
        self.box_predictor_classification = box_predictor_classification

        self.box_predictor_classification.train()
        self.box_head_classification.train()
        optimizer = torch.optim.Adam(
            list(self.box_predictor_classification.parameters()) + list(self.box_head_classification.parameters()),
            lr=float(finetuning_config["learning_rate"]))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=finetuning_config['gamma'])

        # debug reduce dataset for ID 22 where occluded
        # if self.im_index==419:
        #     for t in self.inactive_tracks:
        #         eliminate = 6
        #         print('\n eleminiere die {} letzten von {}'.format(eliminate, t.id))
        #         t.training_set.pos_unique_indices = t.training_set.pos_unique_indices[:(40-eliminate)]
        #         t.training_set.num_frames = 40-eliminate
        # if self.im_index==419:
        #     t = self.inactive_tracks[0]
        #     eliminate = 7
        #     print('\n eleminiere die {} letzten von {}'.format(eliminate,t.id))
        #     t.training_set.pos_unique_indices = t.training_set.pos_unique_indices[:(40-eliminate)]
        #     t.training_set.num_frames = 40-eliminate

        training_set, val_set = self.training_set.get_training_set(self.inactive_tracks, finetuning_config['validate'],
                                                                   finetuning_config['val_split'], finetuning_config['val_set_random'])

        dataloader_train = torch.utils.data.DataLoader(training_set, batch_size=training_set.batch_size, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=training_set.batch_size)

        if self.finetuning_config['early_stopping_classifier']:
            early_stopping = EarlyStopping(patience=self.finetuning_config['early_stopping_patience'], verbose=False, delta=1e-4, checkpoints=self.checkpoints)

        if self.finetuning_config["plot_training_curves"]:
            plotter = VisdomLinePlotter(id=[t.id for t in self.inactive_tracks],
                                        env=self.run_name,
                                        n_samples_train=training_set.max_occ,
                                        n_samples_val=training_set.min_occ,
                                        im=self.im_index)

        # if no val set available, not early stopping possible - make sure to optimize at least 10 epochs
        # if len(val_set) == 0:
        #     train_acc_criterion = True
        if len(val_set) > 0:
            it = int(finetuning_config["epochs"])
        else:
            it = 10

        for i in range(it):
            run_loss = 0.0
            run_acc = 0.0
            for i_sample, sample_batch in enumerate(dataloader_train):
                optimizer.zero_grad()
                loss = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                                 sample_batch['scores'], eval=False)
                #if self.finetuning_config["plot_training_curves"] or len(val_set) == 0:
                if self.finetuning_config["plot_training_curves"]:
                    pred_scores = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                          sample_batch['scores'], eval=True, return_scores=True)
                    mask = torch.argmax(pred_scores, dim=1, keepdim=True).squeeze()
                    corr = torch.sum(mask == sample_batch['scores'])
                    run_acc += 100 * corr.item()
                loss.backward()
                optimizer.step()
                run_loss += loss.detach().item() / len(sample_batch['scores'])

            scheduler.step()
            if finetuning_config["plot_training_curves"]:
                plotter.plot_(epoch=i, loss=run_loss, acc=run_acc / len(dataloader_train.dataset), split_name='train')

            # if train_acc_criterion and (run_acc / len(dataloader_train.dataset)) == 100:
            #     break


            if self.finetuning_config["validate"]:
                run_loss_val = 0.0
                run_acc_val = 0.0
                with torch.no_grad():
                    for i_sample, sample_batch in enumerate(dataloader_val):
                        loss_val = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                                         sample_batch['scores'], eval=True)
                        run_loss_val += loss_val.detach().item() / len(sample_batch['scores'])
                        pred_scores_val = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                              sample_batch['scores'], eval=True, return_scores=True)
                        mask = torch.argmax(pred_scores_val, dim=1, keepdim=True).squeeze()
                        corr = torch.sum(mask == sample_batch['scores'])
                        run_acc_val += 100 * corr.item()

                if finetuning_config["plot_training_curves"] and len(val_set) > 0:
                    plotter.plot_(epoch=i+1, loss=run_loss_val, acc=run_acc_val/len(dataloader_val.dataset), split_name='val')

            if self.finetuning_config['early_stopping_classifier'] and len(val_set) > 0:
                models = [self.box_predictor_classification, self.box_head_classification]
                early_stopping(val_loss=run_loss_val, model=models)
                if early_stopping.early_stop:
                    #print("Early stopping")
                    break

        if self.finetuning_config['early_stopping_classifier'] and self.finetuning_config["validate"] and len(val_set) > 0:
            # load the last checkpoint with the best model
            for i, m in enumerate(models):
                m.load_state_dict(self.checkpoints[i])


        self.box_predictor_classification.eval()
        self.box_head_classification.eval()
        self.num_training += 1
        self.inactive_tracks_temp = self.inactive_tracks.copy()
        self.train_on.append(self.im_index)


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
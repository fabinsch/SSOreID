import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import cv2
from collections import defaultdict
import datetime
import collections
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracktor.track import Track
from tracktor.visualization import plot_compare_bounding_boxes, VisdomLinePlotter, plot_bounding_boxes
from tracktor.utils import bbox_overlaps, warp_pos, get_center, get_height, get_width, make_pos, EarlyStopping, clip_boxes, EarlyStopping2
from tracktor.live_dataset import InactiveDataset, IndividualDataset
from tracktor.training_set_generation import replicate_and_randomize_boxes

from torchvision.ops.boxes import clip_boxes_to_image, nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torch.nn import functional as F
from torchvision.models.detection.transform import resize_boxes
import torch.nn as nn

import time

#if not torch.cuda.is_available():
#    matplotlib.use('TkAgg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg, seq):
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
        self.others_db = {}

        self.inactive_tracks_id = []
        self.inactive_number_changes = 0
        self.box_head_classification = None
        self.box_predictor_classification = None
        self.training_set = None
        now = datetime.datetime.now()
        #self.run_name = now.strftime("%Y-%m-%d %H:%M")
        self.run_name = now.strftime("%Y-%m-%d_%H:%M") + '_' + seq
        self.num_reids = 0
        self.checkpoints = {}
        self.killed_this_step = []
        self.num_training = 0
        self.train_on = []
        self.count_killed_this_step_reid = 0
        self.c_just_one_frame_active = 0
        self.c_skipped_for_train_iou = 0
        self.c_skipped_and_just_and_frame_active = 0
        self.trained_epochs = []
        self.score_others = []

        self.exclude_from_others = []
        self.gt = None
        self.missed_reID = 0
        self.wrong_reID = 0
        self.inactive_tracks_gt_id = []
        self.det_new_track_exclude = []
        self.db_gt_inactive = {}

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        self.inactive_tracks_id = []
        self.inactive_number_changes = 0
        self.num_reids = 0
        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            #t = torch.arange(self.im_index-(t.training_set.features.shape[0]))
            gt_boxes = torch.cat(list(self.gt.values()), 0).to(device)
            inactivetrack_iou_GT = bbox_overlaps(t.pos, gt_boxes).cpu().numpy()
            ind = np.where(inactivetrack_iou_GT == np.max(inactivetrack_iou_GT))[1]
            if len(ind) > 0:
                ind = ind[0]
                overlap = inactivetrack_iou_GT[0, ind]
                if overlap >= 0.5:
                    gt_id = list(self.gt.keys())[ind]
                    t.gt_id = gt_id
                    self.inactive_tracks_gt_id.append(gt_id)

                    if gt_id not in self.db_gt_inactive.keys():
                        self.db_gt_inactive[gt_id] = [t.id]
                    else:
                        self.db_gt_inactive[gt_id].append(t.id)

            if t.frames_since_active > 1:
                t.pos = t.last_pos[-1]
                self.inactive_number_changes += 1
                self.killed_this_step.append(t.id)
                if t.skipped_for_train == 1:
                    self.c_skipped_and_just_and_frame_active += 1
            else:
                self.c_just_one_frame_active += 1
                # remove tracks with just 1 active frame
                #tracks.remove(t)
                t.pos = t.last_pos[-1]
                self.inactive_number_changes += 1
                self.killed_this_step.append(t.id)
                if t.skipped_for_train == 1:
                    self.c_skipped_and_just_and_frame_active += 1
        self.inactive_tracks += tracks


    def add(self, new_det_pos, new_det_scores, image, frame, new_det_features=None):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        old_tracks = self.get_pos()
        box_roi_pool = self.obj_detect.roi_heads.box_roi_pool
        iou = bbox_overlaps(torch.cat((new_det_pos, old_tracks)), torch.cat((new_det_pos, old_tracks)))

        # do augmentation in current frame
        if self.finetuning_config['data_augmentation'] > 0:
            boxes = torch.tensor([]).to(device)
            for i, track in enumerate(new_det_pos):
                box = track
                augmented_boxes = replicate_and_randomize_boxes(box.unsqueeze(0),
                                                                self.finetuning_config['data_augmentation'],
                                                                self.finetuning_config['max_displacement'])
                augmented_boxes = clip_boxes_to_image(augmented_boxes, image.size()[1:3])
                boxes = torch.cat((boxes, torch.cat((box.unsqueeze(0), augmented_boxes))))
        else:
            #boxes = clip_boxes(new_det_pos, image.size()[1:3])
            boxes = new_det_pos

        # do batched roi pooling
        boxes_resized = resize_boxes(boxes, image.size()[1:3], self.obj_detect.image_size[0])
        proposals = [boxes_resized]
        with torch.no_grad():
            roi_pool_feat = box_roi_pool(self.obj_detect.fpn_features, proposals, image.size()[1:3]).to(device)

        roi_pool_per_track = roi_pool_feat.split(self.finetuning_config['data_augmentation'] + 1)
        for i in range(num_new):
            track = Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i,
                          new_det_features[i].view(1, -1) if new_det_features else None, self.inactive_patience, self.max_features_num,
                          self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1,
                          image.size()[1:3], self.obj_detect.image_size, self.finetuning_config["batch_size"],
                          box_roi_pool=box_roi_pool, keep_frames=self.finetuning_config['keep_frames'],
                          data_augmentation = self.finetuning_config['data_augmentation'])
            self.tracks.append(track)
            if frame==138:
                # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
                # debugging frcnn-09 frame 138 problem person wird fälschlicherweise als ID4
                track.add_classifier(self.box_predictor_classification, self.box_head_classification)
                print('\n attached classifier')
            if i in self.det_new_track_exclude:
                self.exclude_from_others.append(track.id)
                print("exclude newly init track {} from others".format(track.id))
            #other_pedestrians_bboxes = torch.cat((new_det_pos[:i], new_det_pos[i + 1:], old_tracks))
            if torch.sum(iou[i] > self.finetuning_config['train_iou_threshold']) > 1:
                print('\nSKIP SKIP SKIP beim Adden')
                self.c_skipped_for_train_iou += 1
                track.skipped_for_train += 1
                continue

            track.update_training_set_classification(features=roi_pool_per_track[i],
                                                     pos=boxes[i+self.finetuning_config['data_augmentation']].unsqueeze(0),
                                                     frame=self.im_index,
                                                     area=track.calculate_area())

        self.track_num += num_new

    def get_box_predictor_(self, n=2):
        if self.finetuning_config['others_class']:
            # Get a box predictor with number of output neurons corresponding to number of inactive tracks + 1 for others
            box_predictor = FastRCNNPredictor(1024, n + 1).to(device)
        else:
            box_predictor = FastRCNNPredictor(1024, n).to(device)
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
            #### count number of parameters #####
            # print(sum(p.numel() for p in box_head.parameters() if p.requires_grad))
        else:
            box_head = self.box_head_classification  # do not start again with pretrained weights

        # for the first time
        if box_head == None:
            box_head =  TwoMLPHead(self.obj_detect.backbone.out_channels *
                                       self.obj_detect.roi_heads.box_roi_pool.output_size[0] ** 2,
                                       representation_size=1024).to(device)
            box_head.load_state_dict(self.bbox_head_weights)
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
                #t.pos = pos[i].view(1, -1)  # here in original implementation
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
            #if self.im_index==117:
             #   tracks=tracks[:-1]
        if len(tracks) == 1:
            pos = tracks[0].pos
        elif len(tracks) > 1:
            pos = torch.cat([t.pos for t in tracks], 0)
        else:
            pos = torch.zeros(0).to(device)
        return pos


    def reid_by_finetuned_model_(self, new_det_pos, new_det_scores, frame, blob):
        """Do reid with one model predicting the score for each inactive track"""
        image = blob['img'][0]
        gt = blob['gt']
        det_gt_id = [] # list which detections corresponds to which gt id
        inactive_tracks_gt_id = []
        current_inactive_tracks_id = [t.id for t in self.inactive_tracks]
        inactive_tracks_gt_id = self.inactive_tracks_gt_id
        samples_per_track = [t.training_set.num_frames_keep for t in self.inactive_tracks]
        assert current_inactive_tracks_id == self.inactive_tracks_id

        #active_tracks = self.get_pos()
        if len(new_det_pos.size()) > 1 and len(self.inactive_tracks) > 0:
            remove_inactive = []
            det_index_to_candidate = defaultdict(list)
            inactive_to_det = defaultdict(list)
            assigned = []
            inactive_tracks = self.get_pos(active=False)

            # for t in self.inactive_tracks:
            #     gt_boxes = torch.cat(list(gt.values()), 0).to(device)
            #     inactivetrack_iou_GT = bbox_overlaps(t.pos, gt_boxes).cpu().numpy()
            #     ind = np.where(inactivetrack_iou_GT == np.max(inactivetrack_iou_GT))[1]
            #     if len(ind) > 0:
            #         ind = ind[0]
            #         overlap = inactivetrack_iou_GT[0, ind]
            #         if overlap >= 0.2:
            #             gt_id = list(gt.keys())[ind]
            #             t.gt_id = gt_id
            #             inactive_tracks_gt_id.append(gt_id)

            for det in new_det_pos:
                gt_boxes = torch.cat(list(gt.values()), 0).to(device)
                det_iou_GT = bbox_overlaps(det.unsqueeze(0), gt_boxes).cpu().numpy()
                ind = np.where(det_iou_GT == np.max(det_iou_GT))[1]
                if len(ind) > 0:
                    ind = ind[0]
                    overlap = det_iou_GT[0, ind]
                    if overlap >= 0.5:
                        det_gt_id.append(list(gt.keys())[ind])
                    else:
                        det_gt_id.append(-1)


            boxes, scores = self.obj_detect.predict_boxes(new_det_pos,
                                                          box_predictor_classification=self.box_predictor_classification,
                                                          box_head_classification=self.box_head_classification,
                                                          pred_multiclass=True)

            zero_scores = [s.item() for s in scores[:,0]] if self.finetuning_config['others_class'] else []
            for zero_score in zero_scores:
                self.score_others.append(zero_score)

            #if frame==420:
            if frame>=0:
                print('\n{}: scores reID: {}'.format(self.im_index, scores))
                print('IDs: {} with respectively {} samples'.format(current_inactive_tracks_id, samples_per_track))

            # check if scores has very high value, don't use IoU restriction in that case
            #no_mask = torch.ge(scores, 0.95)
            # calculate IoU distances
            iou = bbox_overlaps(new_det_pos, inactive_tracks)
            if self.finetuning_config['others_class']:
                # iou has just values for the inactive tracks -> extend for others class
                iou = torch.cat((torch.ones(iou.shape[0],1).to(device), iou), dim=1)
            iou_mask = torch.ge(iou, self.reid_iou_threshold)
            #scores = scores * iou_mask + scores * no_mask



            if self.box_predictor_classification.cls_score.out_features==1:
                scores = scores * iou_mask.squeeze()
                for i, s in enumerate(scores.cpu().numpy()):
                    if s < 0.5:
                        inactive_track = self.inactive_tracks[0]
                        det_index_to_candidate[i].append((inactive_track, s))
                        inactive_to_det[0].append(i)

            else:
                # if self.im_index==138:
                #     iou_mask = torch.cat((iou_mask, torch.tensor([False]).unsqueeze(0).to(device)), dim=1)
                #     iou_mask = torch.cat((iou_mask, torch.tensor([False]).unsqueeze(0).to(device)), dim=1)
                #     #iou_mask = iou_mask
                scores = scores * iou_mask
                scores = scores.cpu().numpy()
                max = scores.max(axis=1)
                max_idx = scores.argmax(axis=1)
                scores[:, max_idx] = 0
                max2 = scores.max(axis=1)
                #max_idx2 = scores.argmax(axis=1)
                dist = max - max2

                for i, d in enumerate(dist):
                    if (max[i] > self.finetuning_config['reid_score_threshold']):
                        if self.finetuning_config['others_class']:
                            # idx = 0 means unknown background people, idx=1,2,.. is inactive
                            if max_idx[i] == 0:
                                print('no reid because class 0 has score {}'.format(max[i]))
                                if det_gt_id[i] in inactive_tracks_gt_id:  # check if the gt id has ever been seen before
                                    f = 0
                                    for t in self.inactive_tracks:  # if it's still in the inactive tracks
                                        if t.gt_id == det_gt_id[i]:
                                            #self.exclude_from_others.append(t.id)
                                            print('!!!!! missed reID of person {}, detection {}'.format(t.id, i))
                                            self.missed_reID += 1
                                            self.det_new_track_exclude.append(i)
                                            f += 1
                                    if f==0:  # means this track is not in inactive anymore but was before
                                        for id in self.db_gt_inactive[det_gt_id[i]]:
                                            self.exclude_from_others.append(id)
                                            print("exlude {}".format(id))

                            else:
                                inactive_track = self.inactive_tracks[max_idx[i] - 1]
                                det_index_to_candidate[i].append((inactive_track, max[i]))
                                inactive_to_det[max_idx[i] - 1].append(i)
                        else:
                            inactive_track = self.inactive_tracks[max_idx[i]]
                            det_index_to_candidate[i].append((inactive_track, max[i]))
                            inactive_to_det[max_idx[i]].append(i)

                    #elif max[i] > 0.0:
                    else:
                        print('no reid with score {}'.format(max[i]))
                        if det_gt_id[i] in inactive_tracks_gt_id:
                            f = 0
                            for t in self.inactive_tracks:
                                if t.gt_id == det_gt_id[i]:
                                    #self.exclude_from_others.append(t.id)
                                    print('!!!!! missed reID of person {}'.format(t.id))
                                    self.missed_reID += 1
                                    self.det_new_track_exclude.append(i)
                                    f += 1
                            if f == 0:
                                for id in self.db_gt_inactive[det_gt_id[i]]:
                                    self.exclude_from_others.append(id)
                                    print("exlude {}".format(id))


            # if frame==13800:
            #     # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
            #     # debugging frcnn-09 frame 138 problem person wird fälschlicherweise als ID4
            #     self.inactive_tracks[max_idx[0]-1].add_classifier(self.box_predictor_classification, self.box_head_classification)
            #     print('\n attached classifier')
            #
            # if frame==15500:
            #     # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
            #     # debugging frcnn-09 frame 138 problem person wird fälschlicherweise als ID4
            #     self.inactive_tracks[max_idx[0]-1].add_classifier(self.box_predictor_classification, self.box_head_classification)
            #     print('\n attached classifier')



            for det_index, candidates in det_index_to_candidate.items():
                candidate = candidates[0]
                inactive_track = candidate[0]
                # get the position of the inactive track in inactive_tracks
                # if just one track, position "is 1" because 0 is unknown background person
                # important for check in next if statement
                inactive_id_in_list = self.inactive_tracks.index(inactive_track)

                if len(inactive_to_det[inactive_id_in_list]) == 1:
                    # check if GT id fits
                    if inactive_track.gt_id != det_gt_id[det_index]:
                        self.exclude_from_others.append(inactive_track.id)
                        print('!!!!! wrong reID of person {}'.format(inactive_track.id))
                        self.wrong_reID += 1
                        print("exclude reID track {} from others".format(inactive_track.id))
                    # make sure just 1 new detection per inactive track
                    self.tracks.append(inactive_track)
                    print(f"\n**************   Reidying track {inactive_track.id} in frame {frame} with score {candidate[1]}")
                    print(' - it was trained on inactive tracks {}'.format([t.id for t in self.inactive_tracks]))
                    self.num_reids += 1

                    if inactive_track.id in self.killed_this_step:
                        self.count_killed_this_step_reid += 1
                        #print('\n track {} was killed and reid in frame {}'.format(inactive_track.id, self.im_index))

                    # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
                    if frame==4200:
                        inactive_track.add_classifier(self.box_predictor_classification, self.box_head_classification)

                    # reset inactive track
                    inactive_track.count_inactive = 0
                    inactive_track.pos = new_det_pos[det_index].view(1, -1)
                    inactive_track.reset_last_pos()
                    inactive_track.skipped_for_train = 0

                    if self.finetuning_config['reset_dataset']:
                        inactive_track.frames_since_active = 1
                        inactive_track.training_set = IndividualDataset(inactive_track.id,
                                                                        self.finetuning_config['keep_frames'],
                                                                        self.finetuning_config['data_augmentation']
                                                                        )

                    assigned.append(det_index)
                    remove_inactive.append(inactive_track)
                else:
                    print('\nerror, {} new det for 1 inactive track ID {}'.format(len(inactive_to_det[inactive_id_in_list]), inactive_track.id))
                    print(' - it was trained on inactive tracks {}'.format([t.id for t in self.inactive_tracks]))

            if len(remove_inactive) > 0:
                # do batched roi pooling
                box_roi_pool = self.obj_detect.roi_heads.box_roi_pool
                if len(remove_inactive) == 1:
                    pos = remove_inactive[0].pos
                else:
                    pos = torch.cat([t.pos for t in remove_inactive], 0)

                # do augmentation in current frame
                if self.finetuning_config['data_augmentation'] > 0:
                    boxes = torch.tensor([]).to(device)
                    for i, track in enumerate(pos):
                        box = track
                        augmented_boxes = replicate_and_randomize_boxes(box.unsqueeze(0), self.finetuning_config['data_augmentation'],
                                                                        self.finetuning_config['max_displacement'])
                        augmented_boxes = clip_boxes_to_image(augmented_boxes, image.shape[-2:])
                        boxes = torch.cat((boxes, torch.cat((box.unsqueeze(0), augmented_boxes))))
                else:
                    #boxes = clip_boxes(pos, image.size()[1:3])
                    boxes = pos

                boxes_resized = resize_boxes(boxes, image.size()[1:3], self.obj_detect.image_size[0])
                proposals = [boxes_resized]
                with torch.no_grad():
                    roi_pool_feat = box_roi_pool(self.obj_detect.fpn_features, proposals, image.size()[1:3]).to(device)
                roi_pool_per_track = roi_pool_feat.split(self.finetuning_config['data_augmentation']+1)

            for i, inactive_track in enumerate(remove_inactive):
                self.inactive_tracks.remove(inactive_track)
                inactive_track.update_training_set_classification(features=roi_pool_per_track[i],
                                                                  pos=boxes[i+self.finetuning_config['data_augmentation']].unsqueeze(0),
                                                                  frame=self.im_index,
                                                                  area=inactive_track.calculate_area())

            keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().to(device)
            if keep.nelement() > 0:
                new_det_pos = new_det_pos[keep]
                new_det_scores = new_det_scores[keep]
            else:
                new_det_pos = torch.zeros(0).to(device)
                new_det_scores = torch.zeros(0).to(device)

        return new_det_pos, new_det_scores


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
        self.gt = blob['gt']
        self.det_new_track_exclude = []

        if self.public_detections:
            dets = blob['dets'].squeeze(dim=0)
            if dets.nelement() > 0:
                boxes, scores = self.obj_detect.predict_boxes(dets)
            else:
                boxes = scores = torch.zeros(0).to(device)
        else:
            boxes, scores = self.obj_detect.detect(blob['img'])

        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

            # Filter out tracks that have too low person score
            inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
        else:
            inds = torch.zeros(0).to(device)

        # Are there any bounding boxes that have a high enough person (class 1) classification score.
        if inds.nelement() > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).to(device)
            det_scores = torch.zeros(0).to(device)

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

                # # calculate IoU distances
                active_pos = self.get_pos()
                # iou = bbox_overlaps(active_pos, active_pos)

                # do augmentation in current frame
                if self.finetuning_config['data_augmentation'] > 0:
                    boxes = torch.tensor([]).to(device)
                    for i, track in enumerate(self.tracks):
                        box = track.pos # shape (1,4)
                        augmented_boxes = replicate_and_randomize_boxes(box, self.finetuning_config['data_augmentation'],
                                                                        self.finetuning_config['max_displacement'])
                        augmented_boxes = clip_boxes_to_image(augmented_boxes, blob['img'].shape[-2:])
                        boxes = torch.cat((boxes, torch.cat((box, augmented_boxes))))
                else:
                    #boxes = clip_boxes(active_pos, blob['img'][0].size()[1:3])
                    # no more clipping needed, as pos in tracks is clipped
                    boxes = active_pos

                # do batched roi pooling
                box_roi_pool = self.obj_detect.roi_heads.box_roi_pool
                boxes_resized = resize_boxes(boxes, blob['img'][0].size()[1:3], self.obj_detect.image_size[0])
                proposals = [boxes_resized]
                with torch.no_grad():
                    roi_pool_feat = box_roi_pool(self.obj_detect.fpn_features, proposals, blob['img'][0].size()[1:3]).to(device)

                roi_pool_per_track = roi_pool_feat.split(self.finetuning_config['data_augmentation']+1)
                for i, track in enumerate(self.tracks):
                    track.frames_since_active += 1

                    # if torch.sum(iou[i] > self.finetuning_config['train_iou_threshold']) > 1:
                    #     self.c_skipped_for_train_iou += 1
                    #     track.skipped_for_train += 1
                    #     continue

                    # debug
                    if hasattr(track, 'box_predictor_classification_debug'):
                        boxes_debug, scores_debug = self.obj_detect.predict_boxes(track.pos, track.box_predictor_classification_debug,track.box_head_classification_debug, pred_multiclass=True)
                        print('\n DEBUG scores track {}: {}'.format(track.id, scores_debug))

                    #track.update_training_set_classification(features=roi_pool_feat[i].unsqueeze(0),
                    track.update_training_set_classification(features=roi_pool_per_track[i],
                                                             pos=boxes[i+self.finetuning_config['data_augmentation']].unsqueeze(0),
                                                             frame=self.im_index,
                                                             area=track.calculate_area())

        # train REID model new if change in active tracks happened
        current_inactive_tracks_id = [t.id for t in self.inactive_tracks]
        if (current_inactive_tracks_id != self.inactive_tracks_id) or self.killed_this_step:
            if self.finetuning_config["for_reid"]:
                box_head_copy_for_classifier = self.get_box_head(reset=self.finetuning_config['reset_head'])  # get head and load weights
                box_predictor_copy_for_classifier = self.get_box_predictor_(n=len(self.inactive_tracks))  # get predictor with corrsponding output number
                # if self.im_index==117:
                #    box_predictor_copy_for_classifier = self.get_box_predictor_(
                #        n=len(self.inactive_tracks)+2)  # get predictor with corrsponding output number

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
                new_det_pos, new_det_scores = self.reid_by_finetuned_model_(new_det_pos, new_det_scores, frame, blob)
            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, blob['img'][0], frame, new_det_features)

        ####################
        # Generate Results #
        ####################

        # calculate IoU distances
        active_pos = self.get_pos()
        iou = bbox_overlaps(active_pos, active_pos)

        for i, t in enumerate(self.tracks):
            if t.id not in self.results.keys():
                self.results[t.id] = {}
                #self.others_db[t.id] = torch.tensor([]).to(device)
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score.cpu()])])

            #if self.finetuning_config['others_class']:
            if self.finetuning_config['others_class']:
                num_others = sum([len(t) for t in self.others_db.values()])
                # make sure, tracks for inactive do not overlap
                if torch.sum(iou[i] > 0.1) > 1 and num_others>160: # make sure to have at least 4 IDs with enough samples
                    #self.others_db[t.id] = (torch.zeros_like(t.training_set.features), t.training_set.frame, torch.zeros(1))
                    # if t.id not in self.others_db.keys():
                    #     self.others_db[t.id] = [
                    #         (0, torch.zeros_like(t.training_set.features), t.training_set.frame)]
                    # else:
                    #     self.others_db[t.id].append(
                    #         (0, torch.zeros_like(t.training_set.features), t.training_set.frame))
                    continue

                #self.others_db[t.id] = (t.training_set.features, t.training_set.frame, t.training_set.area)
                if t.id not in self.others_db.keys():
                    self.others_db[t.id] = [(t.calculate_area(), t.training_set.features[-1], t.training_set.frame[-1])]
                else:
                    self.others_db[t.id].append(
                        (t.calculate_area(), t.training_set.features[-1], t.training_set.frame[-1]))
                #self.others_db[t.id] = t.training_set.features, t.training_set.frame, t.training_set.area)
                self.others_db[t.id].sort(key=lambda tup: tup[0], reverse=True)

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

        # if self.im_index==406:
        #     for t in self.tracks:
        #         if t.id==27:
        #             torch.save(t.training_set.features, 'id27.pt')
        self.im_index += 1
        self.last_image = blob['img'][0]


    def get_results(self):
        return self.results


    def forward_pass_for_classifier_training(self, features, scores, eval=False, return_scores=False, ep=-1, fId=None):
        loss_others = 0
        loss_inactives = 0
        max_sample_loss_others = 0
        if eval:
            self.box_predictor_classification.eval()
            self.box_head_classification.eval()

        # if ep>=99 or ep==0:
        #     activations = {}
        # 
        #     def get_activation(name):
        #         def hook(model, input, output):
        #             activations[name] = output.cpu().detach()
        #
        #         return hook
        #
        #     weights = self.box_head_classification.fc6.weight.data.cpu().numpy()
        #     self.box_head_classification.fc6.register_forward_hook(get_activation('layer0'))

        feat = self.box_head_classification(features)
        # if (ep>=99 or ep==0):
        #     plt.matshow(activations['layer0'])
        #     plt.matshow(weights)
        #     plt.savefig('./training_set/test{}.png'.format(ep))
        class_logits, _ = self.box_predictor_classification(feat)
        if return_scores:
            if self.box_predictor_classification.cls_score.out_features == 1:
                pred_scores = torch.sigmoid(class_logits)
            else:
                pred_scores = F.softmax(class_logits, -1)
            if eval:
                self.box_predictor_classification.train()
                self.box_head_classification.train()
            return pred_scores.detach()
            #return pred_scores[:, 1:].squeeze(dim=1).detach()
        if self.box_predictor_classification.cls_score.out_features==1:
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(class_logits.squeeze(1), scores)
        else:
            loss = F.cross_entropy(class_logits, scores.long())
            if True and ep>=0:
                m = torch.nn.LogSoftmax()
                n = torch.nn.Softmax()
                logscores_each = m(class_logits)
                scores_each = n(class_logits)
                l = torch.nn.NLLLoss(reduction='none')
                loss_each = l(logscores_each, scores.long())
                debug = torch.cat((scores.unsqueeze(1), scores_each), dim=1)
                debug = torch.cat((debug, loss_each.unsqueeze(1)), dim=1)
                debug = torch.cat((debug, fId), dim=1)
                mask = torch.argmax(scores_each, dim=1, keepdim=True).squeeze()
                #correct = torch.sum(mask == scores).item()
                # total = len(scores)
                # if self.im_index == 28 and (ep%50==0):
                #     print("({}.{}) class / scores / loss / frame / ID".format(self.im_index, ep))
                #     print(debug.data.cpu().numpy())
                t, idx, counter = np.unique(scores.cpu().numpy(), return_inverse=True, return_counts=True)
                #counter = collections.Counter(idx)
                for i, c in enumerate(t):
                    p = scores == (torch.ones(scores.shape).to(device)*c)
                    if counter[i] > 0:
                        class_loss = torch.sum(p * loss_each)/counter[i]
                    else:
                        class_loss = -1

                    if c == 0:
                        loss_others = class_loss.detach()
                    else:
                        loss_inactives += torch.sum(p * loss_each.detach())
                    max, ind = torch.max(p * loss_each,dim=0, keepdim=False, out=None)
                    if c == 0:
                        max_sample_loss_others = max.detach()

                    scores_class = scores + ~p * torch.ones(scores.shape).to(device) * (-10000)
                    correct_class = torch.sum(mask == scores_class).item()
                    if counter[i]>0:
                        acc_class = correct_class / counter[i]
                    else:
                        acc_class = -1

                    # if (ep%50)==0:
                    #     print(
                    #         '({}.{}) loss for class {:.0f} is {:.3f}, acc {:.3f} -- max value {:.3f} for (frame, id) {} - scores {}'.format(
                    #             self.im_index,ep, c, class_loss, acc_class, max, fId[ind], scores_each[ind]))

                    # if (31<=ep<=52) or (ep%50==0):
                    #     print('({}.{}) loss for class {:.0f} is {:.3f}, acc {:.3f} -- max value {:.3f} for (frame, id) {} - scores {}'.format(
                    #         self.im_index, ep, c, class_loss.detach(), acc_class, max, fId[ind], scores_each[ind]))
                inactive_samples = len(scores) - counter[0]
                if inactive_samples > 0:
                    loss_inactives /= inactive_samples
                else:
                    loss_inactives = -1

        if eval:
            self.box_predictor_classification.train()
            self.box_head_classification.train()
        if ep>=0:
            return loss, loss_others, loss_inactives, max_sample_loss_others
        else:
            return loss


    def finetune_classification(self, finetuning_config, box_head_classification,
                                box_predictor_classification, early_stopping, killed_this_step):

        # do not train when no tracks
        if len(self.inactive_tracks) == 0:
            self.inactive_tracks_id = [t.id for t in self.inactive_tracks]
            return

        start_time = time.time()
        for t in self.inactive_tracks:
               t.training_set.post_process()
        #print("\n--- %s seconds --- for post process" % (time.time() - start_time))

        self.training_set = InactiveDataset(data_augmentation=self.finetuning_config['data_augmentation'],
                                            others_db=self.others_db,
                                            others_class=self.finetuning_config['others_class'],
                                            im_index=self.im_index,
                                            ids_in_others=self.finetuning_config['ids_in_others'],
                                            val_set_random_from_middle=self.finetuning_config['val_set_random_from_middle'],
                                            exclude_from_others=self.exclude_from_others,
                                            results=self.results)
        self.box_head_classification = box_head_classification
        self.box_predictor_classification = box_predictor_classification

        self.box_predictor_classification.train()
        self.box_head_classification.train()
        if self.finetuning_config['optimizer']=='adam':
            optimizer = torch.optim.Adam(
                list(self.box_predictor_classification.parameters()) + list(self.box_head_classification.parameters()),
                lr=float(finetuning_config["learning_rate"]))
        elif self.finetuning_config['optimizer']=='sgd':
            optimizer = torch.optim.SGD(
                list(self.box_predictor_classification.parameters()) + list(self.box_head_classification.parameters()),
                lr=float(finetuning_config["learning_rate"]))
        else:
            print('\ninvalid optimizer')

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
        start_time = time.time()
        training_set, val_set = self.training_set.get_training_set(self.inactive_tracks, self.tracks, finetuning_config['validate'],
                                                                   finetuning_config['val_split'], finetuning_config['val_set_random'],
                                                                   finetuning_config['keep_frames'])
        #print("\n--- %s seconds --- for getting datasets" % (time.time() - start_time))

        #assert training_set.scores[-1] == len(self.inactive_tracks)
        batch_size = self.finetuning_config['batch_size'] if self.finetuning_config['batch_size'] > 0 else len(training_set)
        #dataloader_train = torch.utils.data.DataLoader(training_set, batch_size=training_set.batch_size, shuffle=True, drop_last=True)
        dataloader_train = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=batch_size) if len(val_set) > 0 else 0

        if self.finetuning_config['early_stopping_classifier']:
            if self.finetuning_config['early_stopping_method'] == 1:
            #early_stopping = EarlyStopping(patience=self.finetuning_config['early_stopping_patience'], verbose=False, delta=1e-4, checkpoints=self.checkpoints)
                early_stopping = EarlyStopping(patience=self.finetuning_config['early_stopping_patience'], verbose=False, delta=1e-3, checkpoints=self.checkpoints)
            else:
                early_stopping = EarlyStopping2(verbose=False, checkpoints=self.checkpoints, ep_safe=int(finetuning_config["epochs_wo_val"]))

        if self.finetuning_config["plot_training_curves"]:
            plotter = VisdomLinePlotter(id=[t.id for t in self.inactive_tracks],
                                        env=self.run_name,
                                        n_samples_train=len(training_set),
                                        #n_samples_train=training_set.max_occ,
                                        n_samples_val=len(val_set),
                                        #n_samples_val=training_set.min_occ,
                                        im=self.im_index)

        # if no val set available, not early stopping possible - make sure to optimize at least 10 epochs
        if len(val_set) > 0:
            ep = int(finetuning_config["epochs"])
        else:
            ep = int(finetuning_config["epochs_wo_val"])

        for i in range(ep):
            if self.finetuning_config["validate"] and len(val_set) > 0:
                #print('\n epoch {}'.format(i))
                run_loss_val = 0.0
                total = 0
                correct = 0
                with torch.no_grad():
                    for i_sample, sample_batch in enumerate(dataloader_val):
                        loss_val, loss_val_others, loss_val_inactive, max_sample_loss_others = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                                         sample_batch['scores'], eval=True, ep=i, fId=sample_batch['frame_id'])
                        #run_loss_val += loss_val.detach().item() / len(sample_batch['scores'])
                        run_loss_val += loss_val.detach().item()
                        pred_scores_val = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                              sample_batch['scores'], eval=True, return_scores=True)

                        if self.box_predictor_classification.cls_score.out_features == 1:
                            mask = torch.ge(pred_scores_val, 0.5)
                            mask = (torch.zeros(pred_scores_val.shape).to(device) + torch.ones(pred_scores_val.shape).to(device) * mask).squeeze(1)
                        else:
                            mask = torch.argmax(pred_scores_val, dim=1, keepdim=True).squeeze()

                        correct += torch.sum(mask == sample_batch['scores']).item()
                        total += len(sample_batch['scores'])

                acc_val = 100 * correct / total
                loss_val = run_loss_val / len(dataloader_val)
                if finetuning_config["plot_training_curves"] and len(val_set) > 0:
                    plotter.plot_(epoch=i, loss=(loss_val, loss_val_others, loss_val_inactive, max_sample_loss_others), acc=acc_val, split_name='val')

            start_time = time.time()
            run_loss = 0.0
            total = 0
            correct = 0
            for i_sample, sample_batch in enumerate(dataloader_train):
                optimizer.zero_grad()
                if self.im_index==360000:
                    loss = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                                     sample_batch['scores'], eval=False,
                                                                     ep=i, fId=sample_batch['frame_id'])
                else:
                    loss = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                                     sample_batch['scores'], eval=False,
                                                                     )
                #if self.finetuning_config["plot_training_curves"] or len(val_set) == 0:
                if self.finetuning_config["plot_training_curves"]:
                    pred_scores = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                          sample_batch['scores'], eval=True, return_scores=True)

                    if self.box_predictor_classification.cls_score.out_features==1:
                        mask = torch.ge(pred_scores, 0.5)
                        mask = (torch.zeros(pred_scores.shape).to(device) + torch.ones(pred_scores.shape).to(device)*mask).squeeze()
                    else:
                        mask = torch.argmax(pred_scores, dim=1, keepdim=True).squeeze()

                    correct += torch.sum(mask == sample_batch['scores']).item()
                loss.backward()
                optimizer.step()
                #run_loss += loss.detach().item() / len(sample_batch['scores'])
                run_loss += loss.detach().item()
                total += len(sample_batch['scores'])

            scheduler.step()
            if finetuning_config["plot_training_curves"] and total > 0:
                plotter.plot_(epoch=i, loss=run_loss / len(dataloader_train), acc=100 * correct / total, split_name='train')

            if self.finetuning_config['early_stopping_classifier'] and len(val_set) > 0:
                models = [self.box_predictor_classification, self.box_head_classification]
                if self.finetuning_config['early_stopping_method']==1 or self.finetuning_config['early_stopping_method']==2:
                    early_stopping(val_loss=loss_val, model=models, epoch=i+1)
                elif self.finetuning_config['early_stopping_method']==3:
                    early_stopping(val_loss=-acc_val, model=models, epoch=i+1)
                elif self.finetuning_config['early_stopping_method']==4:
                    early_stopping(val_loss=loss_val_others, model=models, epoch=i+1)

                if self.finetuning_config['early_stopping_method'] == 1:
                    if early_stopping.early_stop:
                        print("Early stopping after {} epochs".format(early_stopping.epoch))
                        break

            #print("\n--- %s seconds --- for 1 epoch" % (time.time() - start_time))



        if self.finetuning_config['early_stopping_classifier'] and self.finetuning_config["validate"] and len(val_set) > 0:
            # load the last checkpoint with the best model
            self.trained_epochs.append(early_stopping.epoch)
            print("Best model after {} epochs".format(early_stopping.epoch))
            if early_stopping.epoch>5:
                for i, m in enumerate(models):
                    m.load_state_dict(self.checkpoints[i])
            else:
                print('Avoid untrained model after {} epochs, load state after {} epochs'.format(early_stopping.epoch,int(finetuning_config["epochs_wo_val"])))
                for i, m in enumerate(models):
                    m.load_state_dict(early_stopping.checkpoints_250[i])



        self.box_predictor_classification.eval()
        self.box_head_classification.eval()
        self.num_training += 1
        self.inactive_tracks_id = [t.id for t in self.inactive_tracks]
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
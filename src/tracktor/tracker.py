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
import learn2learn as l2l
from learn2learn.algorithms.meta_sgd import meta_sgd_update
from tracktor.utils import tracks_to_inactive_oracle, reid_by_finetuned_model_oracle, add_oracle
#from tracktor.oracle_tracker import OracleTracker

import time
from torch.autograd import grad
import logging

#if not torch.cuda.is_available():
#    matplotlib.use('TkAgg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('main.tracker')


class reID_Model(torch.nn.Module):
    def __init__(self, head, predictor, n):
        super(reID_Model, self).__init__()
        self.head = head
        self.predictor = predictor.cls_score
        self.num_output = n
        self.additional_layer = {}
        if len(n) > 0:
            for i in n:
                if i > 2:
                    self.additional_layer[i] = torch.nn.Linear(1024, i - 2)
                else:
                    self.additional_layer[i] = None
        else:
            self.additional_layer = None

    def forward(self, x, nways):
        feat = self.head(x)
        x = self.predictor(feat)
        if self.additional_layer is not None and nways > 2:
            self.additional_layer[nways].to(device)
            add = self.additional_layer[nways](feat)
            x = torch.cat((x, add), dim=1)
        return x

class Model(torch.nn.Module):  # for meta-sgd-update
    def __init__(self, head, predictor):
        super(Model, self).__init__()
        self.head = head
        self.predictor = predictor

def accuracy(predictions, targets):
    # in target there are first the labels of the tasks (1, N) N*K*2 times, afterwards the labels of others own (0)
    # num_others_own times, than labels for all others (0)
    predictions = predictions.argmax(dim=1).view(targets.shape)
    compare = (predictions == targets)

    non_zero_target = (targets != 0)
    num_non_zero_targets = non_zero_target.sum()

    zero_target = (targets == 0)
    num_zero_targets = zero_target.sum()

    accuracy_task = compare[non_zero_target].sum().float() / num_non_zero_targets if num_non_zero_targets > 0 else torch.zeros(1)
    accuracy_others = compare[zero_target].sum().float() / num_zero_targets

    return accuracy_task.item(), accuracy_others.item()

class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg, seq, reID_weights, ML=False, LR=False,
                 weightening=-1, train_others=True, db=None):
        #self.logger = logging.getLogger('main.tracker')
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
        self.lrs_ml = LR
        self.init_last = self.finetuning_config["init_last"]
        self.LR_per_parameter = False
        if self.finetuning_config["for_tracking"] or self.finetuning_config["for_reid"]:
            # model = reID_Model(head=obj_detect.roi_heads.box_head,
            #                    predictor=obj_detect.roi_heads.box_predictor,
            #                    n=[2])
            #model.load_state_dict(torch.load(reID_weights))
            if ML:
                a = reID_weights
                fixed = True  # fixed version of template neuron

                # when changed to checkpoint that safes optimizer, but some older ones are still working without this if
                if len(a) == 4:
                    logger.info('loaded init after {} epochs'.format(a['epoch']))
                    a = a['state_dict']

                self.bbox_predictor_weights = self.obj_detect.roi_heads.box_predictor.state_dict()  # predictor is last layer
                if fixed:
                    print("fixed version of template")
                    self.bbox_predictor_weights['cls_score.bias'] = a['template_neuron_bias']
                    self.bbox_predictor_weights['cls_score.weight'] = a['template_neuron_weight']
                    if 'others_neuron_weight' in a.keys():
                        print('load others neuron')
                        self.others_neuron_bias = a['others_neuron_bias']
                        self.others_neuron_weight = a['others_neuron_weight']
                else:
                    self.bbox_predictor_weights['cls_score.bias'] = a['module.predictor.bias']
                    self.bbox_predictor_weights['cls_score.weight'] = a['module.predictor.weight']
                    #if not self.finetuning_config['train_others']:
                    if not self.train_others:
                        self.others_neuron_bias = a['module.others_neuron_bias']
                        self.others_neuron_weight = a['module.others_neuron_weight']

                self.bbox_head_weights = self.obj_detect.roi_heads.box_head.state_dict()  # bbox head are the 2 first layers
                self.bbox_head_weights['fc6.bias'] = a['module.head.fc6.bias']
                self.bbox_head_weights['fc6.weight'] = a['module.head.fc6.weight']
                self.bbox_head_weights['fc7.bias'] = a['module.head.fc7.bias']
                self.bbox_head_weights['fc7.weight'] = a['module.head.fc7.weight']

                if LR:
                    # weight, bias, weight, bias, weight, bias
                    # fc6, fc6, cls_score
                    self.lrs = []

                    if len(a) == 7:  # global LR
                        self.lrs.append(a['lrs'].item())
                        logger.info('LR is {}'.format(a['lrs'].item()))
                        self.LR_per_parameter = False
                    else:
                        # LR per parameter
                        self.LR_per_parameter = True
                        if 'others_neuron_weight' not in a.keys():
                            if fixed:
                                for i in range(4):
                                    l = 'module.lrs.' + str(i)
                                    self.lrs.append(a[l])
                                    if len(a[l].shape) == 2:
                                        logger.info('{} weight LR mean is {}'.format(l, a[l].mean()))
                                    else:
                                        logger.info('{} bias LR mean is {}'.format(l, a[l].mean()))
                                # append LR for template neuron
                                self.lrs.append(a['lrs.0'])
                                logger.info('{} weight LR mean is {} [template neuron]'.format('lrs.0', a['lrs.0'].mean()))
                                self.lrs.append(a['lrs.1'])
                                logger.info('{} bias LR mean is {} [template neuron]'.format('lrs.1', a['lrs.1'].mean()))

                            else:
                                for i in range(6):
                                    l = 'lrs.'+str(i)
                                    self.lrs.append(a[l])
                                    # print the mean values
                                    if len(a[l].shape) == 2:
                                        logger.info('{} weight LR mean is {}'.format(l, a[l].mean()))
                                    else:
                                        logger.info('{} bias LR mean is {}'.format(l, a[l].mean()))
                        else:
                            if fixed:
                                self.others_neuron_weight_lr = a['lrs.0']
                                logger.info('{} weight LR mean is {} [others neuron]'.format('lrs.0', a['lrs.0'].mean()))
                                self.others_neuron_bias_lr = a['lrs.1']
                                logger.info('{} bias LR mean is {} [others neuron]'.format('lrs.1', a['lrs.1'].mean()))
                                for i in range(4):
                                    l = 'module.lrs.' + str(i)
                                    self.lrs.append(a[l])
                                    if len(a[l].shape) == 2:
                                        logger.info('{} weight LR mean is {}'.format(l, a[l].mean()))
                                    else:
                                        logger.info('{} bias LR mean is {}'.format(l, a[l].mean()))
                                # append LR for template neuron
                                self.lrs.append(a['lrs.2'])
                                logger.info('{} weight LR mean is {} [template neuron]'.format('lrs.2', a['lrs.2'].mean()))
                                self.lrs.append(a['lrs.3'])
                                logger.info('{} bias LR mean is {} [template neuron]'.format('lrs.3', a['lrs.3'].mean()))
                            else:
                                # get the LR for the others neuron
                                self.others_neuron_weight_lr = a['lrs.0']
                                logger.info('{} weight LR mean is {} [others neuron]'.format('lrs.0', a['lrs.0'].mean()))
                                self.others_neuron_bias_lr = a['lrs.1']
                                logger.info('{} bias LR mean is {} [others neuron]'.format('lrs.1', a['lrs.1'].mean()))
                                for i in range(2,8):
                                    l = 'lrs.' + str(i)
                                    self.lrs.append(a[l])
                                    if len(a[l].shape) == 2:
                                        logger.info('{} weight LR mean is {}'.format(l, a[l].mean()))
                                    else:
                                        logger.info('{} bias LR mean is {}'.format(l, a[l].mean()))
            else:
                self.bbox_predictor_weights = self.obj_detect.roi_heads.box_predictor.state_dict()
                self.bbox_head_weights = self.obj_detect.roi_heads.box_head.state_dict()

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        # either load or use online
        # if db==None:
        #     self.others_db = {}
        # else:
        #     self.others_db = db

        # use both db
        self.train_others = train_others
        self.others_db = {}
        self.others_db_loaded = db
        self.inactive_tracks_id = []
        self.inactive_number_changes = 0
        self.box_head_classification = None
        self.box_predictor_classification = None
        self.training_set = None
        now = datetime.datetime.now()
        self.run_name = now.strftime("%Y-%m-%d_%H:%M") + '_' + seq
        logger.info('if offline saved in: experiments/logs/{}'.format(self.run_name))
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
        self.missed_reID_others = 0
        self.missed_reID_score = 0
        self.missed_reID_score_iou = 0
        self.missed_reID_patience = 0
        self.wrong_reID = 0
        self.correct_reID = 0
        self.correct_no_reID = 0
        self.correct_no_reID_iou = 0
        self.inactive_tracks_gt_id = []
        self.det_new_track_exclude = []
        self.db_gt_inactive = {}
        self.number_made_predictions = 0
        self.inactive_count_succesfull_reID = []
        self.fill_up_to = self.finetuning_config["fill_up_to"]
        self.flexible = []

        ## get statistics
        self.count_nways = {}
        self.count_kshots = {}

        self.acc_after_train = []
        self.acc_val_after_train = []

        self.weightening = weightening
        logger.debug('init done')

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
            self.inactive_number_changes += 1
            self.killed_this_step.append(t.id)
            t.pos = t.last_pos[-1]
            if t.frames_since_active < 1:
                self.c_just_one_frame_active += 1
                # remove tracks with just 1 active frame
                # tracks.remove(t)
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
                          data_augmentation = self.finetuning_config['data_augmentation'], flip_p=self.finetuning_config['flip_p'])
            self.tracks.append(track)
            if frame == 13800:
                # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
                # debugging frcnn-09 frame 138 problem person wird fälschlicherweise als ID4
                track.add_classifier(self.box_predictor_classification, self.box_head_classification)
                print('\n attached classifier')

            # if i in self.det_new_track_exclude:
            #     self.exclude_from_others.append(track.id)
            #     print("exclude newly init track {} from others".format(track.id))

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
            n = n + 1  # Get a box predictor with number of output neurons corresponding to number of inactive tracks + 1 for others
        if type(self.others_db) is tuple:
            k = len(self.others_db[0])
        else:
            k = len(self.others_db)
        if self.finetuning_config['fill_up'] and n<self.fill_up_to and k>=self.fill_up_to:
            n = self.fill_up_to+1 if self.finetuning_config['others_class'] else self.fill_up_to

        box_predictor = FastRCNNPredictor(1024, n).to(device)

        if self.init_last:
            with torch.no_grad():
                if not hasattr(self, 'others_neuron_weight'):
                    # if others neuron was not meta-learned
                    repeated_bias = self.bbox_predictor_weights['cls_score.bias'].repeat(n)
                    repeated_weight = self.bbox_predictor_weights['cls_score.weight'].repeat(n, 1)

                    if self.LR_per_parameter:
                        box_predictor.repeated_bias_lr = self.lrs[5].repeat(n)
                        box_predictor.repeated_weight_lr = self.lrs[4].repeat(n, 1)
                else:
                    # if others neuron also meta-learned
                    repeated_bias = self.bbox_predictor_weights['cls_score.bias'].repeat(n - 1)
                    repeated_bias = torch.cat((self.others_neuron_bias, repeated_bias))
                    repeated_weight = self.bbox_predictor_weights['cls_score.weight'].repeat(n - 1, 1)
                    repeated_weight = torch.cat((self.others_neuron_weight, repeated_weight))

                    if self.LR_per_parameter:
                        box_predictor.repeated_bias_lr = self.lrs[5].repeat(n - 1)
                        box_predictor.repeated_bias_lr = torch.cat((self.others_neuron_bias_lr, box_predictor.repeated_bias_lr))
                        box_predictor.repeated_weight_lr = self.lrs[4].repeat(n - 1, 1)
                        box_predictor.repeated_weight_lr = torch.cat((self.others_neuron_weight_lr, box_predictor.repeated_weight_lr))

                box_predictor.cls_score.bias.data = repeated_bias
                box_predictor.cls_score.weight.data = repeated_weight

        return box_predictor

    def get_box_predictor(self):
        box_predictor = FastRCNNPredictor(1024, 2).to(device)
        box_predictor.load_state_dict(self.bbox_predictor_weights)
        return box_predictor

    def get_box_head(self, reset=True):
        if reset:
            box_head = TwoMLPHead(self.obj_detect.backbone.out_channels *
                                       self.obj_detect.roi_heads.box_roi_pool.output_size[0] ** 2,
                                       representation_size=1024).to(device)
            box_head.load_state_dict(self.bbox_head_weights)
            #### count number of parameters #####
            # print(sum(p.numel() for p in box_head.parameters() if p.requires_grad))
        else:
            box_head = self.box_head_classification  # do not start again with pretrained weights

        # for the first time
        if box_head == None:
            box_head = TwoMLPHead(self.obj_detect.backbone.out_channels *
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
        # if self.im_index > 0:
        #     th = 100
        # else:
        #     th = self.regression_person_thresh
        th = self.regression_person_thresh

        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]

            if scores[i] <= th:
                self.tracks_to_inactive([t])
                #tracks_to_inactive_oracle(self,[t])
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
            logger.debug('no active tracks')
            pos = torch.zeros(0).to(device)
        return pos


    def reid_by_finetuned_model_(self, new_det_pos, new_det_scores, frame, blob):
        """Do reid with one model predicting the score for each inactive track"""
        image = blob['img'][0]
        current_inactive_tracks_id = [t.id for t in self.inactive_tracks]
        samples_per_track = [t.training_set.num_frames_keep for t in self.inactive_tracks]
        assert current_inactive_tracks_id == self.inactive_tracks_id

        #active_tracks = self.get_pos()
        if len(new_det_pos.size()) > 1 and len(self.inactive_tracks) > 0:
            remove_inactive = []
            det_index_to_candidate = defaultdict(list)
            inactive_to_det = defaultdict(list)
            assigned = []
            inactive_tracks = self.get_pos(active=False)

            boxes, scores = self.obj_detect.predict_boxes(new_det_pos,
                                                          box_predictor_classification=self.box_predictor_classification,
                                                          box_head_classification=self.box_head_classification,
                                                          pred_multiclass=True)

            zero_scores = [s.item() for s in scores[:,0]] if self.finetuning_config['others_class'] else []
            for zero_score in zero_scores:
                self.score_others.append(zero_score)

            #if frame==420:
            if frame>=0:
                logger.debug('\n{}: scores reID: {}'.format(self.im_index, scores))
                logger.debug('IDs: {} with respectively {} samples'.format(current_inactive_tracks_id, samples_per_track))

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
                                logger.debug('no reid because class 0 has score {}'.format(max[i]))

                            else:
                                inactive_track = self.inactive_tracks[max_idx[i] - 1]
                                det_index_to_candidate[i].append((inactive_track, max[i]))
                                inactive_to_det[max_idx[i] - 1].append(i)
                        else:  # no others class
                            inactive_track = self.inactive_tracks[max_idx[i]]
                            det_index_to_candidate[i].append((inactive_track, max[i]))
                            inactive_to_det[max_idx[i]].append(i)

                    else:
                        logger.debug('no reid with score {}'.format(max[i]))

            for det_index, candidates in det_index_to_candidate.items():
                candidate = candidates[0]
                inactive_track = candidate[0]
                # get the position of the inactive track in inactive_tracks
                # if just one track, position "is 1" because 0 is unknown background person
                # important for check in next if statement
                inactive_id_in_list = self.inactive_tracks.index(inactive_track)

                if len(inactive_to_det[inactive_id_in_list]) == 1:  # make sure just 1 new detection per inactive track

                    self.tracks.append(inactive_track)
                    logger.debug(f"\n**************   Reidying track {inactive_track.id} in frame {frame} with score {candidate[1]}")
                    logger.debug(' - it was trained on inactive tracks {}'.format([t.id for t in self.inactive_tracks]))
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
                                                                        self.finetuning_config['data_augmentation'],
                                                                        self.finetuning_config['flip_p'])

                    assigned.append(det_index)
                    remove_inactive.append(inactive_track)
                else:
                    logger.debug('\nerror, {} new det for 1 inactive track ID {}'.format(len(inactive_to_det[inactive_id_in_list]), inactive_track.id))
                    logger.debug(' - it was trained on inactive tracks {}'.format([t.id for t in self.inactive_tracks]))

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
                #tracks_to_inactive_oracle(self,[self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

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
                        if track.missed_since<10:
                            boxes_debug, scores_debug = self.obj_detect.predict_boxes(track.pos,
                                                                                      box_predictor_classification=track.box_predictor_classification_debug,
                                                                                      box_head_classification=track.box_head_classification_debug,
                                                                                      pred_multiclass=True)
                            track.following_scores.append(scores_debug.cpu().numpy())
                            track.missed_since += 1
                            if len(track.correct_prediction)==0:
                                logger.debug('not clear what is correct prediction')
                                track.follwing_corr.append(torch.argmax(scores_debug, dim=1).item())
                            elif len(track.correct_prediction)==1:
                                corr = track.correct_prediction[0]
                                max_ind = torch.argmax(scores_debug, dim=1)
                                max_ind = max_ind.item() if scores_debug[:,max_ind].item()>self.finetuning_config['reid_score_threshold'] else -1
                                track.follwing_corr.append(corr == max_ind)
                                # print('\n DEBUG scores track {}|{} : {}'.format(track.id, track.correct_prediction[0]+1,
                                #                                                scores_debug))
                            else:
                                #print('too many inactive tracks with GT id {}'.format(len(track.correct_prediction)))
                                logger.debug('too many inactive tracks with GT id {}'.format(track.gt_id))
                        else:
                            logger.log(5, 'Statistics for track {}|{}'.format(track.id, track.correct_prediction))
                            logger.log(5, track.following_scores)
                            logger.log(5, track.follwing_corr)
                            scores_sum = np.zeros(track.following_scores[0].shape)
                            for s in track.following_scores:
                                scores_sum = np.concatenate((scores_sum, s))
                            logger.log(5, '{} mean scores {}'.format(track.correct_prediction, np.mean(scores_sum[1:], axis=0)))

                            # reset
                            del track.box_predictor_classification_debug


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

                self.finetune_classification(self.finetuning_config,
                                             box_head_copy_for_classifier,
                                             box_predictor_copy_for_classifier,
                                             early_stopping=self.finetuning_config['early_stopping_classifier'],
                                             killed_this_step=self.killed_this_step,
                                             LR=self.lrs_ml)

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
                #new_det_pos, new_det_scores, track_missed = reid_by_finetuned_model_oracle(self, new_det_pos, new_det_scores, frame, blob)
            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, blob['img'][0], frame, new_det_features)
                #add_oracle(self, new_det_pos, new_det_scores, blob['img'][0], frame, track_missed, new_det_features)

        ####################
        # Generate Results #
        ####################

        # calculate IoU distances to make sure tracks for inactive do not overlap
        # active_pos = self.get_pos()
        # if len(active_pos.shape)>1:
        #     iou = bbox_overlaps(active_pos, active_pos)
        # else:
        #     iou = torch.zeros(1,1)

        for i, t in enumerate(self.tracks):
            if t.id not in self.results.keys():
                self.results[t.id] = {}
                #self.others_db[t.id] = torch.tensor([]).to(device)
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score.cpu()])])

            # if (self.finetuning_config['others_class'] or self.finetuning_config['fill_up'] or self.finetuning_config['flexible'])\
            #         and not self.finetuning_config['load_others']:
            if (self.finetuning_config['others_class'] or self.finetuning_config['fill_up'] or self.finetuning_config['flexible']):
            #if True:
            #if False:
                # make sure, tracks for inactive do not overlap
                # num_others = sum([len(t) for t in self.others_db.values()])
                # if torch.sum(iou[i] > 0.1) > 1 and num_others>160: # make sure to have at least 4 IDs with enough samples
                #     #self.others_db[t.id] = (torch.zeros_like(t.training_set.features), t.training_set.frame, torch.zeros(1))
                #     # if t.id not in self.others_db.keys():
                #     #     self.others_db[t.id] = [
                #     #         (0, torch.zeros_like(t.training_set.features), t.training_set.frame)]
                #     # else:
                #     #     self.others_db[t.id].append(
                #     #         (0, torch.zeros_like(t.training_set.features), t.training_set.frame))
                #     self.c_skipped_for_train_iou += 1
                #     continue


                if t.id not in self.others_db.keys():
                    #self.others_db[t.id] = [(t.calculate_area(), t.training_set.features[-1], t.training_set.frame[-1])]
                    self.others_db[t.id] = [(torch.tensor([]), t.training_set.features[-(1 + self.finetuning_config['data_augmentation'])], t.training_set.frame[-1])]
                else:
                    #self.others_db[t.id].append((t.calculate_area(), t.training_set.features[-1], t.training_set.frame[-1]))
                    self.others_db[t.id].append((torch.tensor([]), t.training_set.features[-(1 + self.finetuning_config['data_augmentation'])], t.training_set.frame[-1]))

                # self.others_db[t.id].sort(key=lambda tup: tup[0], reverse=True) # sort according to area
                if len(self.others_db[t.id]) > 10:  # just keep last 40 frames p P
                    self.others_db[t.id] = self.others_db[t.id][:10]

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


    def forward_pass_for_classifier_training(self, features, labels, return_scores=False, ep=-1, fId=None, weights=1):

        feat = self.box_head_classification(features)
        class_logits, _ = self.box_predictor_classification(feat)

        if return_scores:
            if self.box_predictor_classification.cls_score.out_features == 1:
                pred_scores = torch.sigmoid(class_logits)
            else:
                pred_scores = F.softmax(class_logits, -1)

            #loss = F.cross_entropy(class_logits, labels.long(), weight=weights)  # inf is no problem as weight corresponds 0
            loss = F.cross_entropy(class_logits, labels.long(), reduction='none')
            loss = (loss * weights / weights.sum()).sum()  # needs to be sum at the end not mean!

            return pred_scores.detach(), loss


        if self.box_predictor_classification.cls_score.out_features == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(class_logits.squeeze(1), scores)
        else:
            if len(ratio) > 0:
                num_tracks = int(torch.max(labels).item())
                w = torch.ones(num_tracks + 1).to(device)
                w = w * (1 / ratio)
                loss = F.cross_entropy(class_logits, labels.long(), weight=w)
            else:
                loss = F.cross_entropy(class_logits, labels.long())

            # debug
            if True and ep >= 0:
                loss_others = 0
                loss_inactives = 0
                max_sample_loss_others = 0

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

                    if ep==10:
                        print(
                            '({}.{}) loss for class {:.0f} is {:.3f}, acc {:.3f} -- max value {:.3f} for (frame, id) {} - scores {}'.format(
                                self.im_index,ep, c, class_loss, acc_class, max, fId[ind], scores_each[ind]))

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
        if ep >= 0:
            return loss, loss_others, loss_inactives, max_sample_loss_others
        else:
            return loss


    def finetune_classification(self, finetuning_config, box_head_classification,
                                box_predictor_classification, early_stopping, killed_this_step, LR):

        # do not train when no tracks
        if len(self.inactive_tracks) == 0:
            self.inactive_tracks_id = [t.id for t in self.inactive_tracks]
            return

        if len(self.inactive_tracks) not in self.count_nways.keys():
            self.count_nways[len(self.inactive_tracks)] = 1
        else:
            self.count_nways[len(self.inactive_tracks)] += 1

        start_time = time.time()
        for t in self.inactive_tracks:
            t.training_set.post_process()

            if len(t.training_set) not in self.count_kshots.keys():
                self.count_kshots[len(t.training_set)] = 1
            else:
                self.count_kshots[len(t.training_set)] += 1

        #print("\n--- %s seconds --- for post process" % (time.time() - start_time))

        self.training_set = InactiveDataset(data_augmentation=self.finetuning_config['data_augmentation'],
                                            others_db=(self.others_db, self.others_db_loaded),
                                            others_class=self.finetuning_config['others_class'],
                                            im_index=self.im_index,
                                            ids_in_others=self.finetuning_config['ids_in_others'],
                                            val_set_random_from_middle=self.finetuning_config['val_set_random_from_middle'],
                                            exclude_from_others=self.exclude_from_others,
                                            results=self.results,
                                            flip_p=self.finetuning_config['flip_p'],
                                            fill_up=self.finetuning_config['fill_up'],
                                            fill_up_to=self.fill_up_to,
                                            flexible=self.finetuning_config['flexible'],
                                            upsampling=self.finetuning_config['upsampling'],
                                            weightedLoss=self.finetuning_config['weightedLoss'],
                                            samples_per_ID=self.finetuning_config['samples_per_ID'],
                                            train_others=self.train_others,
                                            load_others=self.finetuning_config['load_others'],
                                            weightening=self.weightening)

        self.box_head_classification = box_head_classification
        self.box_predictor_classification = box_predictor_classification

        self.box_predictor_classification.train()
        self.box_head_classification.train()
        if self.finetuning_config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                list(self.box_predictor_classification.parameters()) + list(self.box_head_classification.parameters()),
                lr=float(finetuning_config["learning_rate"]))
        elif self.finetuning_config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                list(self.box_predictor_classification.parameters()) + list(self.box_head_classification.parameters()),
                lr=float(finetuning_config["learning_rate"]))
            if self.lrs_ml:
                # ues the meta-learned per layer learning rates
                if len(self.lrs) > 1:
                    optimizer = torch.optim.SGD(
                        [
                            {"params": self.box_head_classification.fc6.weight, "lr": self.lrs[0]},
                            {"params": self.box_head_classification.fc6.bias, "lr": self.lrs[1]},
                            {"params": self.box_head_classification.fc7.weight, "lr": self.lrs[2]},
                            {"params": self.box_head_classification.fc7.bias, "lr": self.lrs[3]},
                            #{"params": self.box_predictor_classification.cls_score.weight, "lr": self.lrs[4]},
                            {"params": self.box_predictor_classification.cls_score.weight, "lr": box_predictor_classification.repeated_weight_lr},
                            #{"params": self.box_predictor_classification.cls_score.bias, "lr": self.lrs[5]},
                            {"params": self.box_predictor_classification.cls_score.bias, "lr": box_predictor_classification.repeated_bias_lr},
                        ],
                        lr=float(finetuning_config["learning_rate"])
                    )
                else:
                    # one global LR
                    lr = self.lrs[0]
                    optimizer = torch.optim.SGD(
                        list(self.box_predictor_classification.parameters()) + list(
                            self.box_head_classification.parameters()),
                        lr=float(lr))

        else:
            print('\ninvalid optimizer')

        start_time = time.time()
        training_set, val_set = self.training_set.get_training_set(self.inactive_tracks, self.tracks, finetuning_config['validate'],
                                                                   finetuning_config['val_split'], finetuning_config['val_set_random'],
                                                                   finetuning_config['keep_frames'])
        #print("\n--- %s seconds --- for getting datasets" % (time.time() - start_time))
        if self.finetuning_config['flexible']: # test with flexible fill up
            self.box_predictor_classification = self.get_box_predictor_(n=len(self.inactive_tracks)+training_set.flex_num_fill)
            self.flexible=[training_set.flex_num_fill]
            self.box_predictor_classification.train()

            if self.finetuning_config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(
                    list(self.box_predictor_classification.parameters()) + list(
                        self.box_head_classification.parameters()),
                    lr=float(finetuning_config["learning_rate"]))
            elif self.finetuning_config['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(
                    list(self.box_predictor_classification.parameters()) + list(
                        self.box_head_classification.parameters()),
                    lr=float(finetuning_config["learning_rate"]))
                if self.lrs_ml:
                    if len(self.lrs) > 1:
                        optimizer = torch.optim.SGD(
                            [
                                {"params": self.box_head_classification.fc6.weight, "lr": self.lrs[0]},
                                {"params": self.box_head_classification.fc6.bias, "lr": self.lrs[1]},
                                {"params": self.box_head_classification.fc7.weight, "lr": self.lrs[2]},
                                {"params": self.box_head_classification.fc7.bias, "lr": self.lrs[3]},
                                # {"params": self.box_predictor_classification.cls_score.weight, "lr": self.lrs[4]},
                                {"params": self.box_predictor_classification.cls_score.weight,
                                 "lr": box_predictor_classification.repeated_weight_lr},
                                # {"params": self.box_predictor_classification.cls_score.bias, "lr": self.lrs[5]},
                                {"params": self.box_predictor_classification.cls_score.bias,
                                 "lr": box_predictor_classification.repeated_bias_lr},
                            ],
                            lr=float(finetuning_config["learning_rate"])
                        )
                    else:
                        lr = self.lrs[0]
                        optimizer = torch.optim.SGD(
                            list(self.box_predictor_classification.parameters()) + list(
                                self.box_head_classification.parameters()),
                            lr=float(lr))
        #assert training_set.scores[-1] == len(self.inactive_tracks)
        batch_size = self.finetuning_config['batch_size'] if self.finetuning_config['batch_size'] > 0 else len(training_set)
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
                                        n_samples_train_id=(training_set.scores != 0).sum(),
                                        n_samples_train_others=(training_set.scores == 0).sum(),
                                        n_samples_val_id=(val_set.scores != 0).sum() if len(val_set)>0 else 0,
                                        n_samples_val_others=(val_set.scores == 0).sum() if len(val_set)>0 else 0,
                                        im=self.im_index,
                                        offline=self.finetuning_config['plot_offline'])

        # if no val set available, not early stopping possible - make sure to optimize at least 10 epochs
        if len(val_set) > 0:
            ep = int(finetuning_config["epochs"])
        else:
            ep = int(finetuning_config["epochs_wo_val"])

        for i in range(ep):
            if self.finetuning_config["validate"] and len(val_set) > 0:
                #print('\n epoch {}'.format(i))
                run_loss_val = 0.0
                total_val = 0
                correct_val = 0
                loss_val_others = -1
                loss_val_inactive = -1
                max_sample_loss_others = -1
                with torch.no_grad():
                    for i_sample, sample_batch in enumerate(dataloader_val):
                        # loss_val, loss_val_others, loss_val_inactive, max_sample_loss_others = self.forward_pass_for_classifier_training(sample_batch['features'],
                        #                                                  sample_batch['scores'], eval=True, ep=i, fId=sample_batch['frame_id'])
                        predictions, loss_val = self.forward_pass_for_classifier_training(sample_batch['features'], sample_batch['scores'],
                                                                                          return_scores=True, ratio=val_set.occ)

                        #run_loss_val += loss_val.detach().item() / len(sample_batch['scores'])
                        run_loss_val += loss_val.detach().item()
                        acc_val = accuracy(predictions, sample_batch['scores'])

                loss_val = run_loss_val / len(dataloader_val)
                if finetuning_config["plot_training_curves"] and len(val_set) > 0:
                    #plotter.plot_(epoch=i, loss=(loss_val, loss_val_others, loss_val_inactive, max_sample_loss_others), acc=acc_val, split_name='val')
                    plotter.plot_(epoch=i, loss=loss_val, acc=acc_val, split_name='val')

            start_time = time.time()
            run_loss = 0.0

            # get others each iteration
            # if False:
            #     logger.debug('get new others')
            #     training_set = self.training_set.update_others(self.inactive_tracks)
            #     dataloader_train = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

            for i_sample, sample_batch in enumerate(dataloader_train):
                if not self.LR_per_parameter:
                    optimizer.zero_grad()
                if self.im_index >= 5000000:
                    # debug
                    loss = self.forward_pass_for_classifier_training(sample_batch['features'],
                                                                     sample_batch['scores'], eval=False,
                                                                     ep=i, fId=sample_batch['frame_id'])
                else:
                    predictions, loss = self.forward_pass_for_classifier_training(sample_batch['features'], sample_batch['scores'],
                                                                     return_scores=True, weights=sample_batch['weights'])

                if self.finetuning_config["plot_training_curves"] or True:
                    acc = accuracy(predictions, sample_batch['scores'])

                if self.LR_per_parameter:
                    # own way to update parameters to use LR per parameter
                    parameters = list(self.box_head_classification.parameters()) + list(self.box_predictor_classification.parameters())
                    #parameters =  list(torch.nn.Parameter(self.others_neuron_weight.unsqueeze(0)))+list(torch.nn.Parameter(self.others_neuron_bias.unsqueeze(0)))+list(self.box_head_classification.parameters()) + list(self.box_predictor_classification.parameters())
                    diff_params = [p for p in parameters if p.requires_grad]
                    gradients = grad(loss,
                                     diff_params,
                                     allow_unused=True)
                    # for t, g in enumerate(gradients[:5]):
                    #         if len(g.shape) == 2:
                    #             logger.debug('{} grad weight with mean {}'.format(t, g.mean()))
                    #         else:
                    #             logger.debug('{} grad bias with mean {}'.format(t, g.mean()))

                    # get lrs for meta-sgd-update
                    lrs = self.lrs[0:4]  # head
                    lrs.append(self.box_predictor_classification.repeated_weight_lr)
                    lrs.append(self.box_predictor_classification.repeated_bias_lr)

                    model = Model(self.box_head_classification, self.box_predictor_classification)
                    model = meta_sgd_update(model, lrs, gradients)
                    self.box_predictor_classification = model.predictor
                    self.box_head_classification = model.head

                else:
                    loss.backward()
                    optimizer.step()


                run_loss += loss.detach().item()

            #scheduler.step()
            if finetuning_config["plot_training_curves"]:
                plotter.plot_(epoch=i, loss=run_loss / len(dataloader_train), acc=acc, split_name='train')

            if self.finetuning_config['early_stopping_classifier'] and len(val_set) > 0:
                models = [self.box_predictor_classification, self.box_head_classification]
                if self.finetuning_config['early_stopping_method'] == 1 or self.finetuning_config['early_stopping_method'] == 2:
                    early_stopping(val_loss=loss_val, model=models, epoch=i + 1)
                elif self.finetuning_config['early_stopping_method'] == 3:
                    early_stopping(val_loss=-acc_val, model=models, epoch=i + 1)
                elif self.finetuning_config['early_stopping_method'] == 4:
                    early_stopping(val_loss=loss_val_others, model=models, epoch=i + 1)

                if self.finetuning_config['early_stopping_method'] == 1:
                    if early_stopping.early_stop:
                        print("Early stopping after {} epochs".format(early_stopping.epoch))
                        break

            #print("\n--- %s seconds --- for 1 epoch" % (time.time() - start_time))

        logger.debug("\n--- %s seconds --- for training" % (time.time() - start_time))
        if self.finetuning_config['early_stopping_classifier'] and self.finetuning_config["validate"] and len(val_set) > 0:
            # load the last checkpoint with the best model
            self.trained_epochs.append(early_stopping.epoch)
            print("Best model after {} epochs".format(early_stopping.epoch))
            if early_stopping.epoch > 5:
                for i, m in enumerate(models):
                    m.load_state_dict(self.checkpoints[i])
            else:
                print('Avoid untrained model after {} epochs, load state after {} epochs'.format(early_stopping.epoch,int(finetuning_config["epochs_wo_val"])))
                for i, m in enumerate(models):
                    m.load_state_dict(early_stopping.checkpoints_250[i])


        #print('TRAIN ACC {} at the end of fine-tuning 10 steps'.format(correct/total))
        if self.finetuning_config["validate"] and len(val_set) > 0:
            self.acc_val_after_train.append(acc_val)
        self.acc_after_train.append(acc)
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
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.autograd import Variable

import cv2

from .utils import bbox_overlaps, bbox_transform_inv, clip_boxes
from helper.csrc.wrapper.nms import nms

import matplotlib
if not torch.cuda.is_available():
	matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fpn.model.utils.config import cfg as fpn_cfg

class Tracker():
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
		self.motion_model = tracker_cfg['motion_model']

		self.warp_mode = eval(tracker_cfg['warp_mode'])
		self.number_of_iterations = tracker_cfg['number_of_iterations']
		self.termination_eps = tracker_cfg['termination_eps']

		self.reset()

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
			t.pos = t.last_pos
		self.inactive_tracks += tracks

	def add(self, new_det_pos, new_det_scores, new_det_features, im_info):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)

		for i in range(num_new):
			track = Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i,
						  new_det_features[i].view(1, -1), self.inactive_patience, self.max_features_num, im_info)
			track.generate_training_set(plot=False)

			if fpn_cfg.CLASS_AGNOSTIC_BBX_REG:
				RCNN_bbox_pred_copy = nn.Linear(1024, 4)
			else:
				RCNN_bbox_pred_copy = nn.Linear(1024, 4 * self.n_classes)

			RCNN_bbox_pred_copy.load_state_dict(self.obj_detect.RCNN_bbox_pred.state_dict())
			track.finetune_detector(RCNN_bbox_pred_copy, self.obj_detect._PyramidRoI_Feat,
									self.obj_detect._head_to_tail, self.obj_detect.mrcnn_feature_maps, new_det_pos[i])
			self.tracks.append(track)
		self.track_num += num_new

	def regress_tracks(self, blob):
		"""Regress the position of the tracks and also checks their scores."""
		pos = self.get_pos()
		# regress
		_, scores, bbox_pred, rois = self.obj_detect.test_rois(pos)
		if torch.cuda.is_available():
			rois = rois.cuda()
		boxes = bbox_transform_inv(rois, bbox_pred)
		boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
		pos = boxes[:, self.cl*4:(self.cl+1)*4]
		scores = scores[:, self.cl]

		s = []
		for i in range(len(self.tracks)-1,-1,-1):
			t = self.tracks[i]
			t.score = scores[i]
			if scores[i] <= self.regression_person_thresh:
				self.tracks_to_inactive([t])
			else:
				s.append(scores[i])
				# t.prev_pos = t.pos
				t.pos = pos[i].view(1,-1)
		scores_of_active_tracks = torch.Tensor(s[::-1])
		if torch.cuda.is_available():
			scores_of_active_tracks.cuda()
		return scores_of_active_tracks

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			pos = self.tracks[0].pos
		elif len(self.tracks) > 1:
			pos = torch.cat([t.pos for t in self.tracks],0)
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
		new_det_features = self.reid_network.test_rois(blob['app_data'][0], new_det_pos / blob['im_info'][0][2]).data
		if len(self.inactive_tracks) >= 1 and self.do_reid:
			print("LENGTH OF INACTIVE TRACKS IS {}".format((self.inactive_tracks)))
			# calculate appearance distances
			dist_mat = []
			pos = []
			for t in self.inactive_tracks:
				features_list = [t.test_features(feat.view(1, -1)) for feat in new_det_features]
				dist = torch.stack(features_list, 1)
				dist_mat.append(dist)
				pos.append(t.pos)
			if len(dist_mat) > 1:
				dist_mat = torch.cat(dist_mat, 0)
				pos = torch.cat(pos,0)
			else:
				dist_mat = dist_mat[0]
				pos = pos[0]

			# calculate IoU distances
			iou = bbox_overlaps(pos, new_det_pos)
			iou_mask = torch.ge(iou, self.reid_iou_threshold)
			iou_neg_mask = ~iou_mask
			# make all impossible assignemnts to the same add big value
			dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float()*1000
			dist_mat = dist_mat.cpu().numpy()

			row_ind, col_ind = linear_sum_assignment(dist_mat)

			assigned = []
			remove_inactive = []
			for r,c in zip(row_ind, col_ind):
				if dist_mat[r,c] <= self.reid_sim_threshold:
					t = self.inactive_tracks[r]
					self.tracks.append(t)
					t.count_inactive = 0
					t.last_v = torch.Tensor([])
					t.pos = new_det_pos[c].view(1,-1)
					t.add_features(new_det_features[c].view(1,-1))
					assigned.append(c)
					remove_inactive.append(t)

			for t in remove_inactive:
				self.inactive_tracks.remove(t)

			keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long()
			if torch.cuda.is_available():
				keep = keep.cuda()
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

	def clear_inactive(self):
		"""Checks if inactive tracks should be removed."""
		to_remove = []
		for t in self.inactive_tracks:
			if t.is_to_purge():
				to_remove.append(t)
		for t in to_remove:
			self.inactive_tracks.remove(t)

	def get_appearances(self, blob):
		"""Uses the siamese CNN to get the features for all active tracks."""
		new_features = self.reid_network.test_rois(blob['app_data'][0], self.get_pos() / blob['im_info'][0][2]).data
		return new_features

	def add_features(self, new_features):
		"""Adds new appearance features to active tracks."""
		for t,f in zip(self.tracks, new_features):
			t.add_features(f.view(1,-1))

	def align(self, blob):
		"""Aligns the positions of active and inactive tracks depending on camera motion."""
		if self.im_index > 0:
			im1 = self.last_image.cpu().numpy()
			im2 = blob['data'][0][0].cpu().numpy()
			im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
			im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
			sz = im1.shape
			warp_mode = self.warp_mode
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			#number_of_iterations = 5000
			number_of_iterations = self.number_of_iterations
			termination_eps = self.termination_eps
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
			(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria, inputMask=None,
													  gaussFiltSize=1)
			warp_matrix = torch.from_numpy(warp_matrix)
			pos = []
			for t in self.tracks:
				p = t.pos[0]
				p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
				p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

				p1_n = torch.mm(warp_matrix, p1).view(1,2)
				p2_n = torch.mm(warp_matrix, p2).view(1,2)

				pos = torch.cat((p1_n, p2_n), 1)
				if torch.cuda.is_available():
					pos.cuda()

				t.pos = pos.view(1,-1)
				#t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

			if self.do_reid:
				for t in self.inactive_tracks:
					p = t.pos[0]
					p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
					p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)
					p1_n = torch.mm(warp_matrix, p1).view(1,2)
					p2_n = torch.mm(warp_matrix, p2).view(1,2)
					pos = torch. cat((p1_n, p2_n), 1)
					if torch.cuda.is_available():
						pos = torch.cat((p1_n, p2_n), 1).cuda()
					t.pos = pos.view(1,-1)

			if self.motion_model:
				for t in self.tracks:
					if t.last_pos.nelement() > 0:
						p = t.last_pos[0]
						p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
						p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

						p1_n = torch.mm(warp_matrix, p1).view(1,2)
						p2_n = torch.mm(warp_matrix, p2).view(1,2)
						pos = torch.cat((p1_n, p2_n), 1).cuda()

						t.last_pos = pos.view(1,-1)

	def motion(self):
		"""Applies a simple linear motion model that only consideres the positions at t-1 and t-2."""
		for t in self.tracks:
			# last_pos = t.pos.clone()
			# t.last_pos = last_pos
			# if t.last_pos.nelement() > 0:
				# extract center coordinates of last pos

			x1l = t.last_pos[0,0]
			y1l = t.last_pos[0,1]
			x2l = t.last_pos[0,2]
			y2l = t.last_pos[0,3]
			cxl = (x2l + x1l)/2
			cyl = (y2l + y1l)/2

			# extract coordinates of current pos
			x1p = t.pos[0,0]
			y1p = t.pos[0,1]
			x2p = t.pos[0,2]
			y2p = t.pos[0,3]
			cxp = (x2p + x1p)/2
			cyp = (y2p + y1p)/2
			wp = x2p - x1p
			hp = y2p - y1p

			# v = cp - cl, x_new = v + cp = 2cp - cl
			cxp_new = 2*cxp - cxl
			cyp_new = 2*cyp - cyl

			t.pos[0,0] = cxp_new - wp/2
			t.pos[0,1] = cyp_new - hp/2
			t.pos[0,2] = cxp_new + wp/2
			t.pos[0,3] = cyp_new + hp/2

			t.last_v = torch.Tensor([cxp - cxl, cyp - cyl]).cuda()

		if self.do_reid:
			for t in self.inactive_tracks:
				if t.last_v.nelement() > 0:
					# extract coordinates of current pos
					x1p = t.pos[0, 0]
					y1p = t.pos[0, 1]
					x2p = t.pos[0, 2]
					y2p = t.pos[0, 3]
					cxp = (x2p + x1p)/2
					cyp = (y2p + y1p)/2
					wp = x2p - x1p
					hp = y2p - y1p

					cxp_new = cxp + t.last_v[0]
					cyp_new = cyp + t.last_v[1]

					t.pos[0,0] = cxp_new - wp/2
					t.pos[0,1] = cyp_new - hp/2
					t.pos[0,2] = cxp_new + wp/2
					t.pos[0,3] = cyp_new + hp/2

	def step(self, blob):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		for t in self.tracks:
			t.last_pos = t.pos.clone()

		###########################
		# Look for new detections #
		###########################
		self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])
		if self.public_detections:
			dets = blob['dets']

			if len(dets) > 0:
				dets = torch.cat(dets, 0)[:,:4]
				_, scores, bbox_pred, rois = self.obj_detect.test_rois(dets)
			else:
				rois = torch.zeros(0).cuda()
		else:
			_, scores, bbox_pred, rois = self.obj_detect.detect()

		if torch.cuda.is_available():
			rois = rois.cuda()

		if rois.nelement() > 0:
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

			# Filter out tracks that have too low person score
			scores = scores[:, self.cl]
			inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
		else:
			inds = torch.zeros(0).cuda()

        # Are there any bounding boxes that have a high enough person (class 1) classification score.
		if inds.nelement() > 0:
			boxes = boxes[inds]
			det_pos = boxes[:, self.cl*4:(self.cl+1)*4]
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
			if self.motion_model:
				self.motion()
			#regress
			person_scores = self.regress_tracks(blob)

			if len(self.tracks):

				# create nms input
				# new_features = self.get_appearances(blob)

				# nms here if tracks overlap
				emphasized_scores = person_scores.add_(3)
				if torch.cuda.is_available():
					emphasized_scores = emphasized_scores.cuda()
				nms_inp_reg = torch.cat((self.get_pos(), emphasized_scores.view(-1, 1)), 1)
				keep = nms(nms_inp_reg, self.regression_nms_thresh)

				self.tracks_to_inactive([self.tracks[i]
				                         for i in list(range(len(self.tracks)))
				                         if i not in keep])

				if keep.nelement() > 0:
					ones = torch.ones(self.get_pos().size(0)).add_(3).view(-1, 1)
					if torch.cuda.is_available():
						ones = ones.cuda()
					nms_inp_reg = torch.cat((self.get_pos(), ones),1)
					new_features = self.get_appearances(blob)

					self.add_features(new_features)
					num_tracks = nms_inp_reg.size(0)
				else:
					nms_inp_reg = torch.zeros(0).cuda()
					num_tracks = 0

		#####################
		# Create new tracks #
		#####################

		# !!! Here NMS is used to filter out detections that are already covered by tracks. This is
		# !!! done by iterating through the active tracks one by one, assigning them a bigger score
		# !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
		# !!! In the paper this is done by calculating the overlap with existing tracks, but the
		# !!! result stays the same.
		if det_pos.nelement() > 0:
			nms_inp_det = torch.cat((det_pos, det_scores.view(-1,1)), 1)
		else:
			nms_inp_det = torch.zeros(0).cuda()
		if nms_inp_det.nelement() > 0:
			keep = nms(nms_inp_det, self.detection_nms_thresh)
			nms_inp_det = nms_inp_det[keep]
			# check with every track in a single run (problem if tracks delete each other)
			for i in range(num_tracks):
				nms_inp = torch.cat((nms_inp_reg[i].view(1,-1), nms_inp_det), 0)
				keep = nms(nms_inp, self.detection_nms_thresh)
				keep = keep[torch.ge(keep,1)]
				if keep.nelement() == 0:
					nms_inp_det = nms_inp_det.new(0)
					break
				nms_inp_det = nms_inp[keep]

		if nms_inp_det.nelement() > 0:
			new_det_pos = nms_inp_det[:,:4]
			new_det_scores = nms_inp_det[:,4]

			# try to redientify tracks
			print("###Invoke reid####")
			new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

			# add new
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_scores, new_det_features, im_info=blob['im_info'])

		####################
		# Generate Results #
		####################

		for t in self.tracks:
			track_ind = int(t.id)
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			pos = t.pos[0] / blob['im_info'][0][2]
			sc = t.score
			self.results[track_ind][self.im_index] = np.concatenate([pos.cpu().numpy(), np.array([sc])])

		self.im_index += 1
		self.last_image = blob['data'][0][0]

		self.clear_inactive()

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, im_info):
		self.id = track_id
		self.pos = pos
		self.score = score
		self.features = deque([features])
		self.ims = deque([])
		self.count_inactive = 0
		self.inactive_patience = inactive_patience
		self.max_features_num = max_features_num
		self.last_pos = torch.Tensor([])
		self.last_v = torch.Tensor([])
		self.gt_id = None
		self.im_info = im_info
		self.RCNN_bbox_pred = None
		self.head_to_tail = None

	def is_to_purge(self):
		"""Tests if the object has been too long inactive and is to remove."""
		self.count_inactive += 1
		self.last_pos = torch.Tensor([])
		if self.count_inactive > self.inactive_patience:
			return True
		else:
			return False

	def add_features(self, features):
		"""Adds new appearance features to the object."""
		self.features.append(features)
		if len(self.features) > self.max_features_num:
			self.features.popleft()

	def test_features(self, test_features):
		"""Compares test_features to features of this Track object"""
		if len(self.features) > 1:
			features = torch.cat(list(self.features), 0)
		else:
			features = self.features[0]
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features)
		return dist

	def generate_training_set(self, plot=False):
		gt_pos = self.pos
		random_displacement = 3*torch.randn(5, 4)
		if torch.cuda.is_available():
			random_displacement = random_displacement.cuda()
			gt_pos = gt_pos.cuda()
		random_displaced_bboxes = gt_pos.repeat(5, 1) + random_displacement
		self.training_boxes = clip_boxes(random_displaced_bboxes, self.im_info[0][:2])
		print(random_displaced_bboxes)
		print(self.training_boxes)

		if plot:
			rectangles = self.training_boxes.numpy()
			num_rectangles = len(rectangles)
			h, w = self.im_info[0][:2]
			im = np.zeros([int(h.item()), int(w.item())])
			fig, ax = plt.subplots(1)
			ax.imshow(im, cmap='gist_gray_r')
			gt_pos_np = gt_pos.numpy()
			gt_patch = patches.Rectangle((gt_pos_np[0, 0],
									gt_pos[0, 1]),
									gt_pos[0, 2],
									gt_pos[0, 3], linewidth=0.5, edgecolor='r', facecolor='none')
			rects = [patches.Rectangle((rectangles[i, 0],
									rectangles[i, 1]),
									rectangles[i, 2],
									rectangles[i, 3], linewidth=0.5, edgecolor='b', facecolor='none') for i in range(num_rectangles)]
			for i in range(num_rectangles):
				ax.add_patch(rects[i])
			ax.add_patch(gt_patch)

			plt.show()


	def finetune_detector(self, RCNN_bbox_pred, PyramidRoI_Feat, head_to_tail, mrcnn_feature_maps, gt_box):
	#	optimizer = torch.optim.Adam([RCNN_bbox_pred.parameters(), head_to_tail.parameters()], lr=0.0001)
		optimizer = torch.optim.Adam(RCNN_bbox_pred.parameters(), lr=0.0001)
		criterion = torch.nn.SmoothL1Loss()

		rois = self.training_boxes

		padding = torch.zeros(rois.size(0), 1)
		if torch.cuda.is_available():
			padding = padding.cuda()
		rois_padd = torch.cat((padding, rois), 1)
		if torch.cuda.is_available():
			rois.cuda()
			rois_padd = rois_padd.cuda()

		roi_pool_feat = PyramidRoI_Feat(
			mrcnn_feature_maps, rois_padd, self.im_info)

		for i in range(epochs):

			# feed pooled features to top model
			pooled_feat = head_to_tail(roi_pool_feat)

			# compute bbox offset
			bbox_pred = RCNN_bbox_pred(pooled_feat)

			if fpn_cfg.TEST.BBOX_REG:
				# Apply bounding-box regression deltas
				box_deltas = bbox_pred.data
				n_classes = 2
				if fpn_cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
					# Optionally normalize targets by a precomputed mean and stdev
					bbox_stds = torch.FloatTensor(fpn_cfg.TRAIN.BBOX_NORMALIZE_STDS)
					bbox_means = torch.FloatTensor(fpn_cfg.TRAIN.BBOX_NORMALIZE_MEANS)
					if torch.cuda.is_available():
						bbox_stds = bbox_stds.cuda()
						bbox_means = bbox_means.cuda()
					if fpn_cfg.CLASS_AGNOSTIC_BBX_REG:
						box_deltas = box_deltas.view(-1, 4) * bbox_stds + bbox_means
						box_deltas = box_deltas.view(1, -1, 4)
					else:
						box_deltas = box_deltas.view(-1, 4) * bbox_stds + bbox_means
						box_deltas = box_deltas.view(1, -1, 4 * n_classes)

			box_deltas = box_deltas.squeeze(dim=0)

			if torch.cuda.is_available():
				box_deltas = box_deltas.cuda()

			boxes = bbox_transform_inv(rois, bbox_pred)[:, 4:]
			print(box_deltas)
			input(boxes)

			input('forward worked')

			optimizer.zero_grad()
			loss = criterion(boxes, gt_box)
			loss.backward()
			optimizer.step()
			input('backward also worked')

		self.RCNN_bbox_pred = RCNN_bbox_pred
		self.head_to_tail = head_to_tail
		return None

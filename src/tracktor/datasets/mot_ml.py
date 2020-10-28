# from model.test import _get_blobs

from .mot_sequence import MOT17_Sequence
from ..config import get_output_dir

import cv2
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torchvision.transforms import CenterCrop, Normalize, Compose, RandomHorizontalFlip, RandomCrop, ToTensor, RandomResizedCrop
from tracktor.frcnn_fpn import FRCNN_FPN
from torchvision.models.detection.transform import resize_boxes
from tqdm import tqdm
import copy
import random, os
from ..config import cfg


class MOT_ML(MOT17_Sequence):
	"""Multiple Object Tracking Dataset.

	This class builds samples for meta learning. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

	Values for P are normally 18 and K 4
	"""

	def __init__(self, seq_name, split, vis_threshold, P, K, max_per_person, crop_H, crop_W,
				transform, normalize_mean=None, normalize_std=None, build_dataset=True, validation_sequence='MOT17-02'):
		super().__init__(seq_name, vis_threshold=vis_threshold)

		self.data_dir = osp.join(cfg.DATA_DIR, 'MOT_Test')
		self.P = P
		self.K = K
		self.max_per_person = max_per_person
		self.crop_H = crop_H
		self.crop_W = crop_W
		self.build_dataset = build_dataset
		self.val_seq = validation_sequence

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.obj_detect = FRCNN_FPN(num_classes=2).to(self.device)
		self.obj_detect.load_state_dict(torch.load('output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model',
											  map_location=lambda storage, loc: storage))
		self.obj_detect.eval()
		self.name = seq_name

		if transform == "random":
			self.transform = Compose([RandomCrop((crop_H, crop_W)), RandomHorizontalFlip(), ToTensor(), Normalize(normalize_mean, normalize_std)])
		elif transform == "center":
			self.transform = Compose([CenterCrop((crop_H, crop_W)), ToTensor(), Normalize(normalize_mean, normalize_std)])
		else:
			raise NotImplementedError("Tranformation not understood: {}".format(transform))

		self.build_samples()

		if split == 'train':
			pass
		elif split == 'smallTrain':
			self.data = self.data[0::5] + self.data[1::5] + self.data[2::5] + self.data[3::5]
		elif split == 'smallVal':
			self.data = self.data[4::5]
		else:
			raise NotImplementedError("Split: {}".format(split))

	def __getitem__(self, idx):

		# idx belongs to the positive sampled person
		sample = self.data[idx]
		#pers = id[np.random.choice(id.shape[0])]
		id = sample[0]
		box = sample[1]
		feat = sample[2]

		return id, feat

	def build_samples(self):
		"""Builds the samples out of the sequence."""

		tracks = {}

		for j, sample in enumerate(tqdm(self.data)):
			img = Image.open(sample['im_path']).convert("RGB")

			# z = random.randint(0,1000)
			# im_name = '{}_mot.jpg'.format(z)
			# save_path = os.path.join(self.data_dir, 'test_print')
			# im_path = os.path.join(save_path, im_name)
			# if not os.path.isfile(im_path):
			# 	img.save(im_path)

			transform = ToTensor()
			img = transform(img)
			self.obj_detect.load_image(img.unsqueeze(0))
			im_path = sample['im_path']
			gt = sample['gt']
			boxes = torch.tensor(list(gt.values())).to(self.device)
			gt_copy = copy.deepcopy(gt)
			# do roi pooling
			box_roi_pool = self.obj_detect.roi_heads.box_roi_pool
			boxes_resized = resize_boxes(boxes, img.size()[1:3], self.obj_detect.image_size[0])
			proposals = [boxes_resized]
			with torch.no_grad():
				roi_pool_feat = box_roi_pool(self.obj_detect.fpn_features, proposals, self.obj_detect.image_size).to(
					self.device)

			roi_pool_per_track = roi_pool_feat.split(1)

			for k,v in tracks.items():
				if k in gt.keys():
					i = list(gt_copy.keys()).index(k)
					v.append({'id':k, 'im_path':im_path, 'gt':gt[k], 'feat': roi_pool_per_track[i].cpu()})
					del gt[k]

			# For all remaining BB in gt new tracks are created
			for k, v in gt.items():
				i = list(gt_copy.keys()).index(k)
				tracks[k] = [{'id':k, 'im_path':im_path, 'gt':v, 'feat': roi_pool_per_track[i].cpu()}]

			## for debug just to the first N pictures
			# if j > 10:
			# 	break

		# sample max_per_person images and filter out tracks smaller than 4 samples / K samples
		#outdir = get_output_dir("siamese_test")
		#res = []
		pers = []
		for k,v in tracks.items():
			l = len(v)
			if l >= self.K:
				if l > self.max_per_person:
					for i in np.random.choice(l, self.max_per_person, replace=False):
						pers.append((k, v[i]['gt'], v[i]['feat']))
				else:
					for i in range(l):
						pers.append((k, v[i]['gt'], v[i]['feat']))
			else:
				print('\n just {} for ID {}'.format(len(v), k))



		if self._seq_name:
			classes = [c[0] for c in pers]
			print("[*] Loaded {} samples from {} persons in sequence {}.".format(len(pers), len(set(classes)), self._seq_name))

		if len(pers)>1:
			self.data = np.array(pers)
			idx = [person[0] for person in pers]
			box = [person[1] for person in pers]
			feats = [person[2] for person in pers]
			feats = torch.cat(feats)
			#box = np.concatenate(box)
			self.box = box
			self.idx = np.array(idx)
			self.feats = feats
		else:
			self.data = np.array([]).reshape(0,3)

	def build_crop(self, im_path, gt):
		im = cv2.imread(im_path)
		height, width, channels = im.shape
		#blobs, im_scales = _get_blobs(im)
		#im = blobs['data'][0]
		#gt = gt * im_scales[0]
		# clip to image boundary
		w = gt[2] - gt[0]
		h = gt[3] - gt[1]
		context = 0
		gt[0] = np.clip(gt[0]-context*w, 0, width-1)
		gt[1] = np.clip(gt[1]-context*h, 0, height-1)
		gt[2] = np.clip(gt[2]+context*w, 0, width-1)
		gt[3] = np.clip(gt[3]+context*h, 0, height-1)

		im = im[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

		im = cv2.resize(im, (int(self.crop_W*1.125), int(self.crop_H*1.125)), interpolation=cv2.INTER_LINEAR)

		return im

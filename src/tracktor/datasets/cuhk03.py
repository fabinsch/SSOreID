import numpy as np
import cv2
import os
import os.path as osp
import configparser
import csv
import h5py
from PIL import Image, ImageOps
from tracktor.frcnn_fpn import FRCNN_FPN

import torch
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Normalize, Compose, RandomHorizontalFlip, RandomCrop, ToTensor, RandomResizedCrop
from torchvision.models.detection.transform import resize_boxes

from ..config import cfg
from tqdm import tqdm

import random


class CUHK03(Dataset):
	"""CUHK03 dataloader.

	Inspired from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data_manager/cuhk03.py.

	This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

	Values for P are normally 18 and K 4
	"""

	def __init__(self, seq_name, vis_threshold, P, K, max_per_person, crop_H, crop_W,
				transform, normalize_mean=None, normalize_std=None):

		self.data_dir = osp.join(cfg.DATA_DIR, 'cuhk03_release')
		self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')
		if not osp.exists(self.raw_mat_path):
			raise RuntimeError("'{}' is not available".format(self.raw_mat_path))
		self.seq_name = seq_name

		self.P = P
		self.K = K
		self.max_per_person = max_per_person
		self.crop_H = crop_H
		self.crop_W = crop_W

		if transform == "random":
			self.transform = Compose([RandomCrop((crop_H, crop_W)), RandomHorizontalFlip(), ToTensor(), Normalize(normalize_mean, normalize_std)])
		elif transform == "center":
			self.transform = Compose([CenterCrop((crop_H, crop_W)), ToTensor(), Normalize(normalize_mean, normalize_std)])
		else:
			raise NotImplementedError("Tranformation not understood: {}".format(transform))

		if seq_name:
			assert seq_name in ['labeled', 'detected']
			self.data = self.load_images()
		else:
			self.data = []

		self.build_samples()
		

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""Return the ith triplet"""

		res = []
		# idx belongs to the positive sampled person
		pos = self.data[idx]
		res.append(pos[np.random.choice(pos.shape[0], self.K, replace=False)])

		# exclude idx here
		neg_indices = np.random.choice([i for i in range(len(self.data)) if i != idx], self.P-1, replace=False)
		for i in neg_indices:
			neg = self.data[i]
			res.append(neg[np.random.choice(neg.shape[0], self.K, replace=False)])

		# concatenate the results
		r = []
		for pers in res:
			for im in pers:
				im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
				im = Image.fromarray(im)
				r.append(self.transform(im))
		images = torch.stack(r, 0)

		# construct the labels
		labels = [idx] * self.K

		for l in neg_indices:
			labels += [l] * self.K

		labels = np.array(labels)

		batch = [images, labels]

		return batch

	def load_images(self):
		"""Loads the images from the mat file and saves them."""

		#save_path = os.path.join(self.data_dir, self.seq_name)
		#if not osp.exists(save_path):
		#	os.makedirs(save_path)

		mat = h5py.File(self.raw_mat_path, 'r')

		identities = {} # maps entries of the file to identity numbers

		total = []

		def _deref(ref):
			"""Matlab reverses the order of column / row."""
			return mat[ref][:].T

		for campid, camp_ref in enumerate(mat[self.seq_name][0]):
			camp = _deref(camp_ref)     # returns the camera pair
			num_pids = camp.shape[0]    # gets number of identities
			for pid in range(num_pids):
				img_paths = []
				for imgid, img_ref in enumerate(camp[pid,:]):
					img = _deref(img_ref)                      # now we have a single image
					if img.size == 0 or img.ndim < 3: continue # if empty skip
					img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # switch to BGR from RGB

					id_name = '{:01d}_{:03d}'.format(campid+1, pid+1)
					if id_name in identities:
						id_num = identities[id_name]
					else:
						id_num = len(identities)
						identities[id_name] = id_num

					#im_name = '{:04d}_{:01d}.png'.format(id_num, imgid)
					#im_path = os.path.join(save_path, im_name)
					#if not os.path.isfile(im_path):
					#	cv2.imwrite(im_path, img)
					sample = {'im':self.build_crop(img),
							  'id':id_num}
					total.append(sample)

		return total

	def build_samples(self):
		"""Builds the samples for simaese out of the data."""

		tracks = {}

		for sample in self.data:
			im = sample['im']
			identity = sample['id']

			if identity in tracks:
				tracks[identity].append(sample)
			else:
				tracks[identity] = []
				tracks[identity].append(sample)

		# sample max_per_person images and filter out tracks smaller than 4 samples
		#outdir = get_output_dir("siamese_test")
		res = []
		for k,v in tracks.items():
			l = len(v)
			if l >= self.K:
				pers = []
				if l > self.max_per_person:
					for i in np.random.choice(l, self.max_per_person, replace=False):
						pers.append(v[i]['im'])
				else:
					for i in range(l):
						pers.append(v[i]['im'])
				res.append(np.array(pers))

		if self.seq_name:
			print("[*] Loaded {} persons from sequence {}.".format(len(res), self.seq_name))

		self.data = res

	def build_crop(self, im):

		im = cv2.resize(im, (int(self.crop_W*1.125), int(self.crop_H*1.125)), interpolation=cv2.INTER_LINEAR)

		return im


class CUHK03_ML(Dataset):
	"""CUHK03 dataloader.

	Inspired from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data_manager/cuhk03.py.

	This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

	Values for P are normally 18 and K 4
	"""

	def __init__(self, seq_name, vis_threshold, P, K, max_per_person, crop_H, crop_W,
				 transform, normalize_mean=None, normalize_std=None, build_dataset=False, validation_sequence=None):

		self.data_dir = osp.join(cfg.DATA_DIR, 'cuhk03_release')
		self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')
		if not osp.exists(self.raw_mat_path):
			raise RuntimeError("'{}' is not available".format(self.raw_mat_path))
		self.seq_name = seq_name

		self.P = P
		self.K = K
		self.max_per_person = max_per_person
		self.crop_H = crop_H
		self.crop_W = crop_W

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.obj_detect = FRCNN_FPN(num_classes=2).to(self.device)
		self.obj_detect.load_state_dict(torch.load('output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model',
											  map_location=lambda storage, loc: storage))
		self.obj_detect.eval()
		self.name = 'cuhk03'

		# if transform == "random":
		# 	self.transform = Compose([RandomCrop((crop_H, crop_W)), RandomHorizontalFlip(), ToTensor(),
		# 							  Normalize(normalize_mean, normalize_std)])
		# elif transform == "center":
		# 	self.transform = Compose(
		# 		[CenterCrop((crop_H, crop_W)), ToTensor(), Normalize(normalize_mean, normalize_std)])
		# else:
		# 	raise NotImplementedError("Tranformation not understood: {}".format(transform))

		if seq_name:
			assert seq_name in ['labeled', 'detected']
			self.data = self.load_images()
		else:
			self.data = []

		self.build_samples()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""Return the ith triplet"""

		res = []
		# idx belongs to the positive sampled person
		pos = self.data[idx]
		res.append(pos[np.random.choice(pos.shape[0], self.K, replace=False)])

		# exclude idx here
		neg_indices = np.random.choice([i for i in range(len(self.data)) if i != idx], self.P - 1, replace=False)
		for i in neg_indices:
			neg = self.data[i]
			res.append(neg[np.random.choice(neg.shape[0], self.K, replace=False)])

		# concatenate the results
		r = []
		for pers in res:
			for im in pers:
				im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
				im = Image.fromarray(im)
				r.append(self.transform(im))
		images = torch.stack(r, 0)

		# construct the labels
		labels = [idx] * self.K

		for l in neg_indices:
			labels += [l] * self.K

		labels = np.array(labels)

		batch = [images, labels]

		return batch

	def load_images(self):
		"""Loads the images from the mat file and saves them."""

		# save_path = os.path.join(self.data_dir, self.seq_name)
		# if not osp.exists(save_path):
		#	os.makedirs(save_path)

		mat = h5py.File(self.raw_mat_path, 'r')

		identities = {}  # maps entries of the file to identity numbers

		total = []

		def _deref(ref):
			"""Matlab reverses the order of column / row."""
			return mat[ref][:].T

		for campid, camp_ref in enumerate(mat[self.seq_name][0]):  # seq_name = labeled
			camp = _deref(camp_ref)  # returns the camera pair
			num_pids = camp.shape[0]  # gets number of identities
			for pid in range(num_pids):
				img_paths = []
				for imgid, img_ref in enumerate(camp[pid, :]):
					img = _deref(img_ref)  # now we have a single image
					if img.size == 0 or img.ndim < 3: continue  # if empty skip
					img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # switch to BGR from RGB

					id_name = '{:01d}_{:03d}'.format(campid + 1, pid + 1)
					if id_name in identities:
						id_num = identities[id_name]
					else:
						id_num = len(identities)
						identities[id_name] = id_num

					# im_name = '{:04d}_{:01d}.png'.format(id_num, imgid)
					# im_path = os.path.join(save_path, im_name)
					# if not os.path.isfile(im_path):
					#	cv2.imwrite(im_path, img)

					# sample = {'im': self.build_crop(img),
					# 		  'id': id_num}
					# do not crop
					sample = {'im': img,
							  'id': id_num}

					total.append(sample)

		return total

	def build_samples(self):
		"""Builds the samples for simaese out of the data."""

		tracks = {}

		for sample in self.data:
			im = sample['im']
			identity = sample['id']

			if identity in tracks:
				tracks[identity].append(sample)
			else:
				tracks[identity] = []
				tracks[identity].append(sample)

		# sample max_per_person images and filter out tracks smaller than 4 samples
		# outdir = get_output_dir("siamese_test")
		res = []
		pers = []
		num_ids = 0
		for k, v in tqdm(tracks.items()):
			l = len(v)
			if l >= self.K:
				num_ids += 1
				if l > self.max_per_person:
					for i in np.random.choice(l, self.max_per_person, replace=False):
						pers.append((k, self.enocde(v[i]['im']).cpu()))
						#pers.append((k, 0))
				else:
					for i in range(l):
						pers.append((k, self.enocde(v[i]['im']).cpu()))
						#pers.append((k, 0))
				#res.append(np.array(pers))
			# if k == 50:
			# 	break

		if self.seq_name:
			print(f"[*] Loaded {len(pers)} samples from {num_ids} persons from sequence {self.seq_name}.")

		#self.data = res
		if len(pers)>1:
			#self.data = np.array(pers)
			idx = [person[0] for person in pers]
			feats = [person[1] for person in pers]
			feats = torch.cat(feats)
			self.idx = np.array(idx)
			self.feats = feats
		else:
			self.data = np.array([]).reshape(0,3)

	def build_crop(self, im):

		im = cv2.resize(im, (int(self.crop_W*1.125), int(self.crop_H*1.125)), interpolation=cv2.INTER_LINEAR)

		return im

	def padding(self, img, expected_size):
		desired_width = expected_size[0]
		desired_height = expected_size[1]
		delta_width = desired_width - img.size[0]
		delta_height = desired_height - img.size[1]
		pad_width = delta_width // 2
		pad_height = delta_height // 2
		padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
		return ImageOps.expand(img, padding), pad_width, pad_height

	def enocde(self, img):
		#img = Image.open(im_path).convert("RGB")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		org_size = img.size
		img, delta_w, delta_h = self.padding(img, [1400, 800])

		# z = random.randint(0,1000)
		# im_name = '{}_cuhk.jpg'.format(z)
		# save_path = os.path.join(self.data_dir, 'test_print')
		# im_path = os.path.join(save_path, im_name)
		# if not os.path.isfile(im_path):
		# 	img.save(im_path)

		transform = ToTensor()
		img = transform(img)
		self.obj_detect.load_image(img.unsqueeze(0))
		# do roi pooling
		boxes = torch.tensor([delta_w, delta_h, delta_w+float(org_size[0]), delta_h+float(org_size[1])]).unsqueeze(0).to(self.device)
		box_roi_pool = self.obj_detect.roi_heads.box_roi_pool
		boxes_resized = resize_boxes(boxes, img.size()[1:3], self.obj_detect.image_size[0])
		proposals = [boxes_resized]
		with torch.no_grad():
			roi_pool_features = box_roi_pool(self.obj_detect.fpn_features, proposals, self.obj_detect.image_size).to(
				self.device)

		return roi_pool_features
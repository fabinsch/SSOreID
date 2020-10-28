from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import os

from .mot_ml import MOT_ML


class MOT_ML_Wrapper(Dataset):
	"""A Wrapper class for MOT_ML.

	Wrapper class for combining different sequences into one dataset for the MOT_ML
	Dataset.
	"""

	def __init__(self, split, dataloader):

		train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
				'MOT17-11', 'MOT17-13']

		self._dataloader = MOT_ML(None, split=split, **dataloader)
		self.data = []
		self.name = 'mot_ml_wrapper'
		# TODO DEBUG break muss wieder weg
		for j, seq in enumerate(train_folders):
			d = MOT_ML(seq, split=split, **dataloader)
			#for sample in d.data:
			#self._dataloader.data = np.concatenate((self._dataloader.data, d.data))

			# if not os.path.exists('./data/ML_dataset/{}'.format(seq)):
			# 	os.mkdir('./data/ML_dataset/{}'.format(seq))

			self.data.append(d)
			# #comment out because of marcuhmot
			# if os.path.exists('./data/ML_dataset/db_train_2.h5') and j==0:
			# 	os.remove('./data/ML_dataset/db_train_2.h5')
			#
			# if j==0:
			# 	with h5py.File('./data/ML_dataset/db_train_2.h5', 'w') as hf:
			# 		hf.create_dataset('{}/label'.format(seq), data=d.idx)
			# else:
			# 	with h5py.File('./data/ML_dataset/db_train_2.h5', 'a') as hf:
			# 		hf.create_dataset('{}/label'.format(seq), data=d.idx)
			#
			# with h5py.File('./data/ML_dataset/db_train_2.h5', 'a') as hf:
			# 	hf.create_dataset('{}/data'.format(seq), data=d.feats)
			# 	print('keys: {}'.format(hf.keys()))

			# if j == 1:
			# 	break

	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		return self._dataloader[idx]


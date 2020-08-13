from torch.utils.data import Dataset, ConcatDataset
import torch

from .mot_siamese_wrapper import MOT_Siamese_Wrapper
from .cuhk03 import CUHK03, CUHK03_ML
from .market1501 import Market1501, Market1501_ML
from .mot_ml_wrapper import MOT_ML_Wrapper

import time
import os
import h5py

class MarCUHMOT(Dataset):
	"""A Wrapper class that combines Market1501, CUHK03 and MOT16.

	Splits can be used like smallVal, train, smallTrain, but these only apply to MOT16.
	The other datasets are always fully used.
	"""

	def __init__(self, split, dataloader):
		print("[*] Loading Market1501")
		market = Market1501('gt_bbox', **dataloader)
		print("[*] Loading CUHK03")
		cuhk = CUHK03('labeled', **dataloader)
		print("[*] Loading MOT")
		mot = MOT_Siamese_Wrapper(split, dataloader)

		self.dataset = ConcatDataset([market, cuhk, mot])

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]

class MarCUHMOT_ML(Dataset):
	"""A Wrapper class that combines Market1501, CUHK03 and MOT16.

	Splits can be used like smallVal, train, smallTrain, but these only apply to MOT16.
	The other datasets are always fully used.
	"""

	def __init__(self, split, dataloader):

		print("[*] Loading Market1501")
		start_time = time.time()
		market = Market1501_ML('gt_bbox', **dataloader)
		print("--- %s seconds --- for Market1501" % (time.time() - start_time))
		#self.save(market, 'market_test')


		print("[*] Loading CUHK03")
		start_time = time.time()
		cuhk = CUHK03_ML('labeled', **dataloader)
		print("--- %s seconds --- for Cuhk03" % (time.time() - start_time))


		print("[*] Loading MOT")
		start_time = time.time()
		mot = MOT_ML_Wrapper(split, dataloader)
		print("--- %s seconds --- for MOT17" % (time.time() - start_time))


		#self.dataset = ConcatDataset([market, cuhk, mot])
		name = 'marchumot_ML_pad'
		if os.path.exists('./data/ML_dataset/db_train_{}.h5'.format(name)):
			os.remove('./data/ML_dataset/db_train_{}.h5'.format(name))
		self.save([market, cuhk, mot], name)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]

	def save(self, data, name, i=None):
		if i == None:
			i = 0
		if type(data) is list:  # more than 1 database wrapped into one file
			for d in data:
				self.save(d, name, i)
				i += 1
		else:
			if data.name == 'mot_ml_wrapper':
				for d in data.data:
					self.save(d, name, i)
			else:
				if i == 0:
					with h5py.File('./data/ML_dataset/db_train_{}.h5'.format(name), 'w') as hf:
						hf.create_dataset('{}/label'.format(data.name), data=data.idx)
				else:
					with h5py.File('./data/ML_dataset/db_train_{}.h5'.format(name), 'a') as hf:
						hf.create_dataset('{}/label'.format(data.name), data=data.idx)

				with h5py.File('./data/ML_dataset/db_train_{}.h5'.format(name), 'a') as hf:
					hf.create_dataset('{}/data'.format(data.name), data=data.feats)
					print('keys: {}'.format(hf.keys()))
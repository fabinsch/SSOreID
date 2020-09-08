from torch.utils.data import Dataset, ConcatDataset
import torch

from .mot_reid_wrapper import MOTreIDWrapper
from .cuhk03 import CUHK03
from .market1501 import Market1501


class MarCUHMOT(Dataset):
	"""A Wrapper class that combines Market1501, CUHK03 and MOT16.

	Splits can be used like smallVal, train, smallTrain, but these only apply to MOT16.
	The other datasets are always fully used.
	"""

	def __init__(self, split, dataloader, MOT_val_seq):
		print("[*] Loading Market1501")
		market = Market1501('gt_bbox', **dataloader)
		print("[*] Loading CUHK03")
		cuhk = CUHK03('labeled', **dataloader)
		print("[*] Loading MOT")
		mot = MOTreIDWrapper(split, dataloader, val_seq=MOT_val_seq)

		self.dataset = ConcatDataset([market, cuhk, mot])

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]


class MarCUH(Dataset):
	"""A Wrapper class that combines Market1501, CUHK03

	Splits can be used like smallVal, train, smallTrain, but these only apply to MOT16.
	The other datasets are always fully used.
	"""

	def __init__(self, split, dataloader, MOT_val_seq):
		print("[*] Loading Market1501")
		market = Market1501('gt_bbox', **dataloader)
		print("[*] Loading CUHK03")
		cuhk = CUHK03('labeled', **dataloader)

		self.dataset = ConcatDataset([market, cuhk])

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]

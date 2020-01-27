import torch
from torch import randperm
from torch.utils.data import Subset
from collections import defaultdict
from torchvision.ops.boxes import box_iou

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id):
        self.id = id
        self.features = torch.tensor([]).to(device)
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.samples_per_frame = None
        self.number_of_positive_examples = None

    def append_samples(self, training_set_dict):
        self.features = torch.cat((self.features, training_set_dict['features']))
        self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        self.scores = torch.cat((self.scores, training_set_dict['scores']))

    # Filter out all duplicates and add frame number tensor for each data point
    NUMBER_OF_POSITIVE_EXAMPLE_DUPLICATES = 15
    def post_process(self):
        print("post processing data")
        self.samples_per_frame = defaultdict(list)
        unique_indices = []
        number_of_duplicates = 0
        frame_number = 0
        current_box = self.boxes[0, :]
        for i, box in enumerate(torch.cat((self.boxes[1:, :], torch.Tensor([[0,0,0,0]]).to(device)))):
            if not torch.equal(box, current_box):
                if number_of_duplicates == self.NUMBER_OF_POSITIVE_EXAMPLE_DUPLICATES:
                    frame_number += 1
                # TODO Sort by proximity to first member in list
                unique_indices.append(i)
                self.samples_per_frame[frame_number].append(len(unique_indices) - 1)
                current_box = box
                number_of_duplicates = 0
            else:
                number_of_duplicates += 1
        self.scores = self.scores[torch.LongTensor(unique_indices)]
        self.boxes = self.boxes[torch.LongTensor(unique_indices), :]
        self.features = self.features[torch.LongTensor(unique_indices),:,:,:]
        self.number_of_positive_examples = self.scores[self.scores==1].size()[0]
        assert len(self.samples_per_frame) == self.number_of_positive_examples
        self.sort_by_iou()
        print("DONE")

    def sort_by_iou(self):
        for frame_number in self.samples_per_frame:
            box_idx_in_frame = self.samples_per_frame[frame_number]
            pos_box = self.boxes[box_idx_in_frame[0], :].unsqueeze(0)
            ious = []
            for box_idx in box_idx_in_frame:
                iou = box_iou(self.boxes[box_idx, :].unsqueeze(0), pos_box)
                ious.append(iou.cpu().numpy()[0][0])
            box_idx_in_frame_sorted = [index for iou, index  in sorted(zip(ious, box_idx_in_frame), key=lambda x: x[0], reverse=True)]
            assert len(box_idx_in_frame) == len(box_idx_in_frame_sorted)
            assert box_idx_in_frame[0] == box_idx_in_frame_sorted[0]
            self.samples_per_frame[frame_number] = box_idx_in_frame_sorted

    def val_test_split(self, num_frames_train=20, num_frames_val=10, train_val_frame_gap=0, downsampling=True, upsampling=False):
        assert num_frames_train + num_frames_val + train_val_frame_gap <= self.number_of_positive_examples, \
            "There are not enough frames in the data set"
        pos_idx_train = []
        neg_idx_train = []
        for frame_number in range(num_frames_train):
            pos_idx_train.append(self.samples_per_frame[frame_number+1][0])
            if downsampling:
                # Choose the box as negative example that is closest to the positive example box
                neg_idx_train.append(self.samples_per_frame[frame_number+1][1])
            if upsampling:
                break

        pos_idx_val = []
        neg_idx_val = []
        for frame_number in range(num_frames_train + train_val_frame_gap, num_frames_train + train_val_frame_gap + num_frames_val):
            pos_idx_val.append(self.samples_per_frame[frame_number+1][0])
            if downsampling:
                # Choose the box as negative example that is closest to the positive example box
                neg_idx_val.append(self.samples_per_frame[frame_number+1][1])
            if upsampling:
                break

        train_idx = torch.cat((torch.LongTensor(pos_idx_train), torch.LongTensor(neg_idx_train)))
        val_idx = torch.cat((torch.LongTensor(pos_idx_val), torch.LongTensor(neg_idx_val)))
        return [Subset(self, train_idx), Subset(self, val_idx)]

    def val_test_split_old(self, percentage_positive_examples_train=None, number_positive_examples_train=None,
                       ordered_by_frame=False, downsample=True):
        if percentage_positive_examples_train is None and number_positive_examples_train is None:
            raise ValueError("either a percentage or absolute number of positive examples should be provided")
        if number_positive_examples_train is not None and number_positive_examples_train >= self.number_of_positive_examples:
            raise ValueError("the number of positive samples you provided exceeds the number of available samples")
        if percentage_positive_examples_train is not None and number_positive_examples_train is not None:
            import warnings
            warnings.warn("you provided a percentage and an absolute number. giving preference to percentage")
        if percentage_positive_examples_train is not None:
            train_positive_examples = int(self.number_of_positive_examples * percentage_positive_examples_train)
        else:
            train_positive_examples = number_positive_examples_train
        positive_examples_indices = (self.scores==1).nonzero().squeeze(1)
        negative_examples_indices = (self.scores==0).nonzero().squeeze(1)
        if not ordered_by_frame:
            positive_examples_indices = positive_examples_indices[randperm(positive_examples_indices.size()[0]).tolist()]
            negative_examples_indices = negative_examples_indices[randperm(negative_examples_indices.size()[0]).tolist()]

        # Do downsampling? or upsampling?
        # ..Downsample
        if downsample:
            train_idx = torch.cat((positive_examples_indices[:train_positive_examples],
                                   negative_examples_indices[:train_positive_examples]))
            val_idx = positive_examples_indices[train_positive_examples:]
            val_idx = torch.cat((val_idx,
                                negative_examples_indices[-val_idx.size()[0]:]))

        return [Subset(self, train_idx), Subset(self, val_idx)]

    def establish_class_balance(self):
        return

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}
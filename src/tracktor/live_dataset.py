import torch
from torch import randperm
from torch.utils.data import Subset
from collections import defaultdict
from torchvision.ops.boxes import box_iou

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id, batch_size):
        self.id = id
        self.batch_size = batch_size
        self.number_positive_duplicates = self.batch_size / 2 - 1
        self.features = torch.tensor([]).to(device)
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.samples_per_frame = None
        self.number_of_positive_examples = None
        self.keep_frames = 40
        self.num_frames = 0

        self.pos_unique_indices = []

    def append_samples(self, training_set_dict):
        self.num_frames += 1
        self.features = torch.cat((self.features, training_set_dict['features']))
        self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        self.scores = torch.cat((self.scores, training_set_dict['scores']))
        if self.num_frames > self.keep_frames:
            self.remove_samples()

    def remove_samples(self):
        self.boxes = self.boxes[self.batch_size:, :]
        self.scores = self.scores[self.batch_size:]
        self.features = self.features[self.batch_size:, :, :, :]

    # Filter out all duplicates and add frame number tensor for each data point
    def post_process(self):
        self.samples_per_frame = defaultdict(list)
        unique_indices = []
        number_of_duplicates = 0
        frame_number = 0
        current_box = self.boxes[0, :]
        saw_positive = False
        pos_unique_indices = []
        for i, box in enumerate(torch.cat((self.boxes[1:, :], torch.Tensor([[0,0,0,0]]).to(device)))):
            if not torch.equal(box, current_box):
                if number_of_duplicates == self.number_positive_duplicates and not saw_positive:  # every image again 31 times duplicates of this person
                    frame_number += 1
                    saw_positive = True  # changes to true after the first filtering out of positive duplicates
                    pos_unique_indices.append(i)
                else:
                    saw_positive = False
                unique_indices.append(i)  # first i = 31, next 42, 53
                self.samples_per_frame[frame_number].append(i)  # frame number and the corresponding unique ind per track dataset
                current_box = box
                number_of_duplicates = 0
            else:
                number_of_duplicates += 1
        unique_scores = self.scores[unique_indices]
        self.number_of_positive_examples = unique_scores[unique_scores==1].size()[0]
        assert len(self.samples_per_frame) == self.number_of_positive_examples  # check if in every frame is the positive sample
        self.sort_by_iou()
        self.pos_unique_indices = pos_unique_indices

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

    def get_training_set(self):
        num_train = 40 if self.number_of_positive_examples > 40 else self.number_of_positive_examples
        training_set, _ = self.val_test_split(num_frames_train=num_train, num_frames_val=0, train_val_frame_gap=0,
                                              downsampling=False)

        return training_set, training_set

    def val_test_split(self, num_frames_train=20, num_frames_val=10, train_val_frame_gap=0, downsampling=True,
                       shuffle=True):
        assert num_frames_train + num_frames_val + train_val_frame_gap <= self.number_of_positive_examples, \
            "There are not enough frames in the data set"
        pos_idx_train = []
        neg_idx_train = []
        pos_idx_val = []
        neg_idx_val = []
        if train_val_frame_gap == 0 and num_frames_val == 0:
            for i in range(self.number_of_positive_examples - num_frames_train - num_frames_val, self.number_of_positive_examples - num_frames_val):
                pos_idx_for_frame = self.samples_per_frame[i + 1][0]
                pos_idx_train.append(pos_idx_for_frame)
                if downsampling:
                    # Choose the box as negative example that has highest iou with positive example box
                    neg_idx_train.append(self.samples_per_frame[i + 1][1])
                else:
                    neg_idx_for_frame = self.samples_per_frame[i + 1][1:]
                    repeated_pos_frame = [pos_idx_for_frame] * (len(neg_idx_for_frame) - 1)
                    pos_idx_train.extend(repeated_pos_frame)
                    neg_idx_train.extend(neg_idx_for_frame)

            for i in range(self.number_of_positive_examples - num_frames_val, self.number_of_positive_examples):
                pos_idx_for_frame = self.samples_per_frame[i + 1][0]
                pos_idx_val.append(pos_idx_for_frame)
                if downsampling:
                    # Choose the box as negative example that has highest iou with positive example box
                    neg_idx_val.append(self.samples_per_frame[i + 1][1])
                else:
                    neg_idx_for_frame = self.samples_per_frame[i + 1][1:]
                    repeated_pos_frame = [pos_idx_for_frame] * (len(neg_idx_for_frame) - 1)
                    pos_idx_val.extend(repeated_pos_frame)
                    neg_idx_val.extend(neg_idx_for_frame)
            train_idx = torch.cat((torch.LongTensor(pos_idx_train), torch.LongTensor(neg_idx_train)))
            val_idx = torch.cat((torch.LongTensor(pos_idx_val), torch.LongTensor(neg_idx_val)))
            return  [Subset(self, train_idx), Subset(self, val_idx)]

        for frame_number in range(num_frames_train):
            pos_idx_for_frame = self.samples_per_frame[frame_number + 1][0]
            pos_idx_train.append(pos_idx_for_frame)
            if downsampling:
                # Choose the box as negative example that has highest iou with positive example box
                neg_idx_train.append(self.samples_per_frame[frame_number + 1][1])
            else:
                neg_idx_for_frame = self.samples_per_frame[frame_number + 1][1:]
                repeated_pos_frame = [pos_idx_for_frame] * (len(neg_idx_for_frame) - 1)
                pos_idx_train.extend(repeated_pos_frame)
                neg_idx_train.extend(neg_idx_for_frame)

        for frame_number in range(num_frames_train + train_val_frame_gap, num_frames_train + train_val_frame_gap + num_frames_val):
            pos_idx_for_frame = self.samples_per_frame[frame_number+1][0]
            pos_idx_val.append(pos_idx_for_frame)
            if downsampling:
                # Choose the box as negative example that has highest iou with positive example box
                neg_idx_val.append(self.samples_per_frame[frame_number+1][1])
            else:
                neg_idx_for_frame = self.samples_per_frame[frame_number + 1][1:]
                repeated_pos_frame = [pos_idx_for_frame] * (len(neg_idx_for_frame) - 1)
                pos_idx_val.extend(repeated_pos_frame)
                neg_idx_val.extend(neg_idx_for_frame)

        train_idx = torch.cat((torch.LongTensor(pos_idx_train), torch.LongTensor(neg_idx_train)))
        val_idx = torch.cat((torch.LongTensor(pos_idx_val), torch.LongTensor(neg_idx_val)))
        if shuffle:
            train_idx = train_idx[randperm(len(train_idx))]
            val_idx = val_idx[randperm(len(val_idx))]
        return [Subset(self, train_idx), Subset(self, val_idx)]

    def establish_class_balance(self):
        return

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}


class InactiveDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.features = torch.tensor([]).to(device)
        self.max_occ = 0

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}

    def generate_ind(self, pos_unique_indices, max_occ):
        diff = max_occ - len(pos_unique_indices)
        for i in range(diff):
            pos_unique_indices.append(random.choice(pos_unique_indices))
        return pos_unique_indices

    def get_training_set(self, inactive_tracks):
        occ = [t.training_set.number_of_positive_examples for t in inactive_tracks]
        self.max_occ = max(occ) if len(occ) > 0 else 0
        cl = 0
        # get a random dataset with label 0 if just one inactive track
        if len(inactive_tracks) == 1:
            t = inactive_tracks[0]
            neg_idx = []
            for f in range(self.max_occ):
                neg_idx.append(random.choice(t.training_set.samples_per_frame[f+1][1:]))
            self.scores = torch.zeros(len(neg_idx)).to(device)
            self.boxes = t.training_set.boxes[neg_idx]
            self.features = t.training_set.features[neg_idx]
            cl = 1

        for i, t in enumerate(inactive_tracks):
            # balance dataset, same number of examples for each class
            if len(t.training_set.pos_unique_indices) < self.max_occ:
                pos_unique_indices = self.generate_ind(t.training_set.pos_unique_indices, self.max_occ)
            else:
                pos_unique_indices = t.training_set.pos_unique_indices
            self.scores = torch.cat((self.scores, t.training_set.scores[pos_unique_indices] * (cl+i)))
            self.boxes = torch.cat((self.boxes, t.training_set.boxes[pos_unique_indices]))
            self.features = torch.cat((self.features, t.training_set.features[pos_unique_indices]))

        return self, self


import torch
from torch import randperm
from torch.utils.data import Subset
from collections import defaultdict
from torchvision.ops.boxes import box_iou

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id, batch_size, keep_frames):
        self.id = id
        #self.batch_size = batch_size
        #self.number_positive_duplicates = self.batch_size / 2 - 1
        self.features = torch.tensor([]).to(device)
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.samples_per_frame = None
        #self.number_of_positive_examples = None
        self.keep_frames = keep_frames
        self.num_frames = 0
        self.pos_unique_indices = None

    def append_samples(self, training_set_dict):
        self.num_frames += 1
        self.features = torch.cat((self.features, training_set_dict['features']))
        self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        self.scores = torch.cat((self.scores, training_set_dict['scores']))
        if self.num_frames > self.keep_frames:
            self.remove_samples()

    def remove_samples(self):
        # remove the first i entries (this is where a new pos samples starts)
        i = (self.scores==1).nonzero()[1].item()
        self.boxes = self.boxes[i:, :]
        self.scores = self.scores[i:]
        self.features = self.features[i:, :, :, :]

    # Filter out all duplicates and add frame number tensor for each data point
    def post_process(self):
        self.samples_per_frame = defaultdict(list)
        self.pos_unique_indices = []
        frame_number = -1
        for i, s in enumerate(self.scores):
            if s == 1:
                self.pos_unique_indices.append(i)
                frame_number += 1
            self.samples_per_frame[frame_number].append(i)


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
        num_train = self.number_of_positive_examples
        if num_train > 40 :
            print('More than 40 positive examples')
        #num_train = 40 if self.number_of_positive_examples > 40 else self.number_of_positive_examples
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
        self.min_occ = 0

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}

    def generate_ind(self, pos_unique_indices, max_occ):
        diff = max_occ - len(pos_unique_indices)
        for i in range(diff):
            pos_unique_indices.append(random.choice(pos_unique_indices))
        return pos_unique_indices

    def get_val_idx(self, occ, inactive_tracks, split):
        """generates indices for validation set und removes them from training set"""
        val_idx = []
        self.min_occ = min(occ) if len(occ) > 0 else 0
        for t in inactive_tracks:
            idx = []
            num_val = int(self.min_occ * split)
            for i in range(num_val):
                # take random samples
                # random.shuffle(t.training_set.pos_unique_indices)
                # idx.append(t.training_set.pos_unique_indices.pop())

                # take samples from middle of scene to avoid taking the last occluded ones
                pos_ind = t.training_set.pos_unique_indices
                idx_val = (int(len(pos_ind) / 2) + int(num_val * 0.5))
                idx.append(pos_ind.pop(idx_val))
            val_idx.append(idx)
        return val_idx

    def get_val_set(self, val_idx, inactive_tracks):
        val_set = InactiveDataset(batch_size=64)
        cl = 0
        # get a random dataset with label 0 if just one inactive track
        if len(inactive_tracks) == 1:
            t = inactive_tracks[0]
            neg_idx = []
            f = int(len(t.training_set.samples_per_frame) / 2) - int(len(val_idx[0])/2)
            for i in range(len(val_idx[0])):
                if len(t.training_set.samples_per_frame[f+i][1:]) > 0:
                    neg_idx.append(random.choice(t.training_set.samples_per_frame[f+i][1:]))
                else:
                    other_pers = []
                    # problem if there is no other person in the same frame
                    for j in range(len(t.training_set.samples_per_frame)):
                        if len(t.training_set.samples_per_frame[j][1:]) > 0:
                            other_pers.extend(t.training_set.samples_per_frame[j][1:])
                    for k in range(len(val_idx[0]) - f):
                            neg_idx.append(random.choice(other_pers))
                    break

            val_set.scores = torch.zeros(len(neg_idx)).to(device)
            val_set.boxes = t.training_set.boxes[neg_idx]
            val_set.features = t.training_set.features[neg_idx]
            cl = 1

        for i, idxs in enumerate(val_idx):
            t = inactive_tracks[i]
            val_set.scores = torch.cat((val_set.scores, t.training_set.scores[idxs] * (cl + i)))
            val_set.boxes = torch.cat((val_set.boxes, t.training_set.boxes[idxs]))
            val_set.features = torch.cat((val_set.features, t.training_set.features[idxs]))

        return val_set


    def get_training_set(self, inactive_tracks, val, split):
        val_idx = [[]]
        #occ = [t.training_set.num_frames for t in inactive_tracks]
        occ = [t.training_set.num_frames if t.training_set.num_frames < t.training_set.keep_frames else t.training_set.keep_frames for t in inactive_tracks]
        self.max_occ = max(occ) if len(occ) > 0 else 0

        if val:
            val_idx = self.get_val_idx(occ, inactive_tracks, split)
            self.max_occ -= len(val_idx[0])

        cl = 0
        # get a random dataset with label 0 if just one inactive track
        if len(inactive_tracks) == 1:
            t = inactive_tracks[0]
            neg_idx = []
            i = list(t.training_set.samples_per_frame.keys())[0]
            for f in range(self.max_occ):
                if val:
                    neg_idx.append(random.choice(t.training_set.samples_per_frame[f+i][1:]))
                    t.training_set.samples_per_frame[f+i].remove(neg_idx[-1])
                else:
                    neg_idx.append(random.choice(t.training_set.samples_per_frame[f+i][1:]))
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

        if val:
            val_set = self.get_val_set(val_idx, inactive_tracks)
            return self, val_set

        return self, self


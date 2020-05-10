import torch
from torch import randperm
from torch.utils.data import Subset, ConcatDataset
from collections import defaultdict
from torchvision.ops.boxes import box_iou
import itertools


import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id, batch_size, keep_frames, data_augmentation):
        self.id = id
        #self.batch_size = batch_size
        #self.number_positive_duplicates = self.batch_size / 2 - 1
        self.features = torch.tensor([]).to(device)
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.samples_per_frame = None
        #self.number_of_positive_examples = None
        self.keep_frames = keep_frames
        self.num_frames_keep = 0
        self.num_frames = 0
        self.pos_unique_indices = None
        self.data_augmentation = data_augmentation if data_augmentation > 0 else 1

    def append_samples(self, training_set_dict):
        self.num_frames += 1
        self.num_frames_keep += 1
        self.features = torch.cat((self.features, training_set_dict['features']))
        self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        self.scores = torch.cat((self.scores, torch.tensor([0]).float().to(device)))
        #self.pos_unique_indices = list(range(self.num_frames_keep))
        if self.num_frames > self.keep_frames:
            self.remove_samples()
            self.num_frames_keep = self.keep_frames

    def remove_samples(self):
        self.boxes = self.boxes[1:, :]
        self.scores = self.scores[1:]
        self.features = self.features[1:, :, :, :]

    # add frame number tensor for each data point
    # WEGLASSEN ? ist es notwendig samples per frame zu haben ?
    def post_process(self):
        # self.samples_per_frame = defaultdict(list)
        # self.pos_unique_indices = []
        # frame_number = 0
        # c = 1
        # saw_negative = False  # TODO was ist wenn es keine negative gib ? kann das vorkommen oder ist dann dummy ?
        # for i, s in enumerate(self.scores):
        #     if s == 1 and c <= self.data_augmentation:
        #         self.pos_unique_indices.append(i)
        #         c += 1
        #         if saw_negative:
        #             frame_number += 1
        #             saw_negative = False
        #     else:
        #         if not saw_negative:
        #             c = 1
        #             saw_negative = True
        #
        #     self.samples_per_frame[frame_number].append(i)
        self.pos_unique_indices = list(range(self.num_frames_keep))


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
    def __init__(self, batch_size, killed_this_step=[]):
        self.batch_size = batch_size
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.features = torch.tensor([]).to(device)
        self.max_occ = 0
        self.min_occ = 0
        self.killed_this_step = killed_this_step

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}

    def generate_ind(self, indices, occ):
        if len(indices) > 0:
            diff = occ - len(indices)
            for i in range(diff):
                indices.append(random.choice(indices))
            return indices
        else:
            return []

    def get_others(self, num, tracks, concat_dataset=None, possible_persons=[], split=0, val=False, ):
        if len(tracks) > 0:
            idx = []
            if concat_dataset == None:
                tracks_dataset = [t.training_set for t in tracks]
                concat_dataset = ConcatDataset(tracks_dataset)
                num_possible_persons = len(concat_dataset)
                possible_persons = list(range(num_possible_persons))
            else:
                num_possible_persons = len(possible_persons)

            if val:
                num = num if num < int(num_possible_persons * split) else int(num_possible_persons * split)
                random.shuffle(possible_persons)
                idx = possible_persons[:num]
                possible_persons = possible_persons[num+1:]
            else:
                num = num if num < num_possible_persons else num_possible_persons
                random.shuffle(possible_persons)
                idx = possible_persons[:num]

            return torch.tensor(idx), concat_dataset, possible_persons

        else:
            return torch.tensor([]), concat_dataset, possible_persons


    def get_current_idx(self, num, inactive_tracks, newest_inactive, split=0, val=False):
        idx=[]
        t = [t for t in inactive_tracks if t.id in self.killed_this_step]
        if len(t) == 0:
            return
        # elif len(t) == 1:
        #     t = t[0]
        # else:
        #     t = random.choice(t)[0]
        else:
            # take frame with higher number of frames since active
            active = [track.frames_since_active for track in t]
            index_max = max(range(len(active)), key=active.__getitem__)
            t = t[index_max]
        newest_inactive = newest_inactive if newest_inactive < len(t.training_set.samples_per_frame) else len(t.training_set.samples_per_frame)
        #d = t.training_set.data_augmentation if t.training_set.data_augmentation > 0 else 1
        d = t.training_set.data_augmentation
        if newest_inactive == 0:
            # take all other persons from all frames as samples for others
            num_possible_persons = sum([len(t.training_set.samples_per_frame[f][d:]) for f in t.training_set.samples_per_frame])
            if val:
                num = num if num < int(num_possible_persons*split) else int(num_possible_persons*split)
            else:
                num = num if num < num_possible_persons else num_possible_persons
            for i in range(num):
                while True:
                    f = random.choice(range(len(t.training_set.samples_per_frame)))
                    if len(t.training_set.samples_per_frame[f][d:]) == 0:
                        pass
                    else:
                        idx.append(random.choice(t.training_set.samples_per_frame[f][d:]))
                        t.training_set.samples_per_frame[f].remove(idx[-1])
                        break

        else:
            last = []
            for i in range(1, newest_inactive):
                last.append(list(t.training_set.samples_per_frame.keys())[-i])
            num_possible_persons = sum([len(t.training_set.samples_per_frame[f][d:]) for f in last])
            possible_persons = [t.training_set.samples_per_frame[f][d:] for f in last]
            possible_persons = list(itertools.chain.from_iterable(possible_persons))
            if val:
                num = num if num < int(num_possible_persons*split) else int(num_possible_persons*split)
            else:
                num = num if num < num_possible_persons else num_possible_persons
            for i in range(num):
                random.shuffle(possible_persons)
                idx.append(possible_persons.pop())
        return idx, t

    def get_val_idx(self, occ, inactive_tracks, tracks, split, val_set_random):
        """generates indices for validation set und removes them from training set"""
        val_idx = []
        self.min_occ = min(occ) if len(occ) > 0 else 0
        for t in inactive_tracks:
            idx = []
            num_val = int(self.min_occ * split)
            #num_val = 1 if (num_val==0 and self.min_occ>1) else num_val
            for i in range(num_val):
                if val_set_random:
                    # take random samples
                    random.shuffle(t.training_set.pos_unique_indices)
                    idx.append(t.training_set.pos_unique_indices.pop())
                else:
                    # take samples from middle of scene to avoid taking the last occluded ones
                    pos_ind = t.training_set.pos_unique_indices
                    idx_val = (int(len(pos_ind) / 2) + int(num_val * 0.5))
                    idx.append(pos_ind.pop(idx_val))
            val_idx.append(idx)

        others_idx, others_dataset, pp = self.get_others(num_val, tracks, split=split, val=True)
        #val_others_this_step, t_val = self.get_current_idx(num_val, inactive_tracks, newest_inactive, split=split, val=True)  # append random idx of person that was visible in the last frame
        #val_others_this_step, t_val = self.get_others_idx(num_val, inactive_tracks, tracks, split=split, val=True)  # append random idx of person that was visible in the last frame
        #if len(val_others_this_step) < num_val:
         #   val_others_this_step = self.generate_ind(val_others_this_step, num_val)
        #return val_idx, val_others_this_step, t_val
        return val_idx, others_idx, others_dataset, pp

    def get_val_set(self, val_idx, val_others_idx, inactive_tracks, other_dataset):
        val_set = InactiveDataset(batch_size=64, killed_this_step=self.killed_this_step)

        if len(val_others_idx) > 0:
            val_set.scores = torch.zeros(len(val_others_idx)).to(device)
            for i in val_others_idx:
                val_set.boxes = torch.cat((val_set.boxes, other_dataset[i.long()]['boxes'].unsqueeze(0)))
                val_set.features = torch.cat((val_set.features, other_dataset[i.long()]['features'].unsqueeze(0)))

        for i, idxs in enumerate(val_idx):
            t = inactive_tracks[i]
            val_set.scores = torch.cat((val_set.scores, torch.ones(len(idxs)).to(device) * (i+1)))
            val_set.boxes = torch.cat((val_set.boxes, t.training_set.boxes[idxs]))
            val_set.features = torch.cat((val_set.features, t.training_set.features[idxs]))

        return val_set


    def get_training_set(self, inactive_tracks, tracks, val, split, val_set_random, keep_frames):
        val_idx = [[]]
        if len(self.killed_this_step) == 0:
            self.killed_this_step.append(inactive_tracks[-1].id)  # if no track was killed, take others from newest inactive
        #occ = [t.training_set.num_frames for t in inactive_tracks]
        occ = [t.training_set.num_frames if t.training_set.num_frames < keep_frames else keep_frames for t in inactive_tracks]
        self.max_occ = max(occ) if len(occ) > 0 else 0

        # check when last time a track was added before this newest killed one
        # inactive_since = [t.count_inactive for t in inactive_tracks[:-1]]
        # newest_inactive = min(inactive_since) if len(inactive_since) > 0 else 0

        if val:
            #val_idx, val_others, t_val = self.get_val_idx(occ, inactive_tracks, split, newest_inactive, val_set_random)
            val_idx, val_others_idx, others_dataset, pp = self.get_val_idx(occ, inactive_tracks, tracks, split, val_set_random)
            self.max_occ -= len(val_idx[0])

        # get a random dataset with label 0 if just one inactive track
        train_others_idx, others_dataset, _ = self.get_others(self.max_occ, tracks, concat_dataset=others_dataset, possible_persons=pp)
        if len(train_others_idx) < self.max_occ:
            #train_others = self.generate_ind(train_others_idx, self.max_occ)
            if len(train_others_idx) == 0:
                print('\nkeine other tracks , nicht zu augmenten')
            print('\naugment')
        self.scores = torch.zeros(len(train_others_idx)).to(device)
        # self.boxes = t_train.training_set.boxes[train_others]
        # self.features = t_train.training_set.features[train_others]
        for i in train_others_idx:
            self.boxes = torch.cat((self.boxes, others_dataset[i.long()]['boxes'].unsqueeze(0)))
            self.features = torch.cat((self.features, others_dataset[i.long()]['features'].unsqueeze(0)))

        for i, t in enumerate(inactive_tracks):
            # balance dataset, same number of examples for each class
            if t.training_set.num_frames < self.max_occ:
                pos_unique_indices = list(range(t.training_set.num_frames)) if t.training_set.num_frames < keep_frames else list(range(keep_frames))
                pos_unique_indices = self.generate_ind(pos_unique_indices, self.max_occ)
            else:
                pos_unique_indices = list(range(t.training_set.num_frames)) if t.training_set.num_frames < keep_frames else list(range(keep_frames))
            self.scores = torch.cat((self.scores, torch.ones(len(pos_unique_indices)).to(device) * (i+1)))
            self.boxes = torch.cat((self.boxes, t.training_set.boxes[pos_unique_indices]))
            self.features = torch.cat((self.features, t.training_set.features[pos_unique_indices]))

        if val:
            val_set = self.get_val_set(val_idx, val_others_idx, inactive_tracks, others_dataset)
            return self, val_set

        return self, self


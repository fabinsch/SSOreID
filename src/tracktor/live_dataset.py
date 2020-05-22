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
        #self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.samples_per_frame = None
        #self.number_of_positive_examples = None
        self.keep_frames = keep_frames
        self.num_frames_keep = 0
        self.num_frames = 0
        self.pos_unique_indices = None
        #self.data_augmentation = data_augmentation if data_augmentation > 0 else 1
        self.data_augmentation = data_augmentation

    def append_samples(self, training_set_dict):
        self.num_frames += 1
        self.num_frames_keep += 1
        self.features = torch.cat((self.features, training_set_dict['features']))
        #self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        self.scores = torch.cat((self.scores, torch.tensor([0]).float().to(device)))
        #self.pos_unique_indices = list(range(self.num_frames_keep))
        if self.num_frames > self.keep_frames:
            self.remove_samples()
            self.num_frames_keep = self.keep_frames

    def remove_samples(self):
        #self.boxes = self.boxes[1:, :]
        self.scores = self.scores[1:]
        self.features = self.features[1+self.data_augmentation:, :, :, :]

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
        return self.scores.size()[0]

    def __getitem__(self, idx):
        #return {'features': self.features[idx:(idx+self.data_augmentation+1), :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]*torch.ones(self.data_augmentation+1)}
        intervall = self.data_augmentation + 1  # has to be jumped to get corresponding features
        return {'features': self.features[idx*intervall:(idx*intervall+intervall), :, :, :], 'scores': self.scores[idx]*torch.ones(self.data_augmentation+1)}


class InactiveDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, data_augmentation=0):
        self.batch_size = batch_size
        #self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.features = torch.tensor([]).to(device)
        self.max_occ = 0
        self.min_occ = 0
        #self.killed_this_step = killed_this_step
        self.data_augmentation = data_augmentation

    def __len__(self):
        return self.scores.size()[0]

    def __getitem__(self, idx):
        #return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}
        return {'features': self.features[idx, :, :, :], 'scores': self.scores[idx]}

    def generate_ind(self, indices, occ):
        if len(indices) > 0:
            diff = occ - len(indices)
            for i in range(diff):
                indices.append(random.choice(indices))
            return indices
        else:
            return []

    def get_others(self, num, tracks, concat_dataset=None, c_tracks=0, split=0, val=False, num_val=0):
        current_num = 0
        num_tracks = len(tracks)
        if num_tracks > 0:

            num_frames = [t.frames_since_active for t in tracks]
            num_frames_total = sum(num_frames) - num_val

            if val:
                if num_tracks > 1:
                    s = 0
                    c_tracks = 0
                    num = num if num < int(num_frames_total * split) else int(num_frames_total * split)
                    for n in num_frames[::-1]:
                        current_num += n
                        c_tracks += 1
                        if current_num >= num:
                            e = c_tracks
                            break
                else:
                    return [], concat_dataset, c_tracks, current_num


            else:
                num = num if num < num_frames_total else num_frames_total
                s = c_tracks
                e = None

            tracks_dataset = [t.training_set for t in tracks][::-1]
            concat_dataset = ConcatDataset(tracks_dataset[s:e])

            return list(range(num)), concat_dataset, c_tracks, current_num

        else:
            return [], concat_dataset, c_tracks, current_num

        #     if concat_dataset == None:
        #         tracks_dataset = [t.training_set for t in tracks]
        #         concat_dataset = ConcatDataset(tracks_dataset)
        #         num_possible_persons = len(concat_dataset)
        #         possible_persons = list(range(num_possible_persons))
        #     else:
        #         num_possible_persons = len(possible_persons)
        #
        #     if val:
        #         num = num if num < int(num_possible_persons * split) else int(num_possible_persons * split)
        #         random.shuffle(possible_persons)
        #         idx = possible_persons[:num]
        #         possible_persons = possible_persons[num+1:]
        #     else:
        #         num = num if num < num_possible_persons else num_possible_persons
        #         random.shuffle(possible_persons)
        #         idx = possible_persons[:num]
        #
        #     return idx, concat_dataset, possible_persons
        #
        # else:
        #     return [], concat_dataset, possible_persons


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
        exclude_for_val = []
        just_one_sample_for_val = []
        org_min_occ = min(occ) if len(occ) > 0 else 0
        while True:
            min_occ = min(occ) if len(occ) > 0 else 0
            if min_occ >= 5 or min_occ == 0:
                if min_occ == 1000:
                    self.min_occ = org_min_occ
                else:
                    self.min_occ = min_occ
                break
            else:
                # this would results in not creating a validation set
                idx_min = occ.index(min_occ)
                occ[idx_min] = 1000  # high number to keep index right
                if min_occ == 1:
                    print("\no validation sample for {}".format(idx_min))
                    exclude_for_val.append(idx_min)
                else:
                    just_one_sample_for_val.append(idx_min)

        for i, t in enumerate(inactive_tracks):
            idx = []
            num_val = int(self.min_occ * split)
            if i in exclude_for_val:
                num_val = 0
            if i in just_one_sample_for_val:
                num_val = 1
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

        num_val = int(self.min_occ * split)
        others_idx, others_dataset, c_tracks, num_exclude = self.get_others(num_val, tracks, split=split, val=True)
        #if len(val_others_this_step) < num_val:
         #   val_others_this_step = self.generate_ind(val_others_this_step, num_val)
        return val_idx, num_val, others_idx, others_dataset, c_tracks, num_exclude

    def get_val_set(self, val_idx, val_others_idx, inactive_tracks, other_dataset):
        val_set = InactiveDataset(batch_size=64)

        if len(val_others_idx) > 0:
            val_set.scores = torch.zeros(len(val_others_idx)*(self.data_augmentation+1)).to(device)
            for i in val_others_idx:
                #val_set.boxes = torch.cat((val_set.boxes, other_dataset[i]['boxes'].unsqueeze(0)))
                val_set.features = torch.cat((val_set.features, other_dataset[i]['features']))

        for i, idxs in enumerate(val_idx):
            idx_features = self.expand_indices_augmentation(idxs)
            t = inactive_tracks[i]
            val_set.scores = torch.cat((val_set.scores, torch.ones(len(idx_features)).to(device) * (i+1)))
            #val_set.boxes = torch.cat((val_set.boxes, t.training_set.boxes[idxs]))
            val_set.features = torch.cat((val_set.features, t.training_set.features[idx_features]))

        return val_set

    def expand_indices_augmentation(self, indices):
        idx = []
        inv = self.data_augmentation+1
        for i in indices:
            idx.extend(list(range(i*inv,i*inv+inv)))
        return idx


    def get_training_set(self, inactive_tracks, tracks, val, split, val_set_random, keep_frames):
        val_idx = [[]]
        others_dataset = None
        c_tracks = 0
        num_exclude = 0
        # if len(self.killed_this_step) == 0:
        #     self.killed_this_step.append(inactive_tracks[-1].id)  # if no track was killed, take others from newest inactive
        #occ = [t.training_set.num_frames for t in inactive_tracks]
        #occ = [t.training_set.num_frames_keep if t.training_set.num_frames_keep>1 else -1 for t in inactive_tracks]
        occ = [t.training_set.num_frames_keep for t in inactive_tracks]
        self.max_occ = max(occ) if len(occ) > 0 else 0

        # get idx of validation samples
        if val:
            val_idx, num_val, val_others_idx, others_dataset, c_tracks, num_exclude = self.get_val_idx(occ, inactive_tracks, tracks, split, val_set_random)
            self.max_occ -= num_val

        # get a random dataset with label 0 if just one inactive track
        train_others_idx, others_dataset, _, _ = self.get_others(self.max_occ, tracks, concat_dataset=others_dataset, c_tracks=c_tracks, num_val=num_exclude)
        if len(train_others_idx) < self.max_occ:
            train_others = self.generate_ind(train_others_idx, self.max_occ)
            if len(train_others_idx) == 0:
                print('\nkeine other tracks , nicht zu augmenten')
            print('\naugment')
        self.scores = torch.zeros(len(train_others_idx)*(self.data_augmentation+1)).to(device)
        for i in train_others_idx:
            #self.boxes = torch.cat((self.boxes, others_dataset[i]['boxes'].unsqueeze(0)))
            self.features = torch.cat((self.features, others_dataset[i]['features']))

        for i, t in enumerate(inactive_tracks):
            # balance dataset, same number of examples for each class
            if len(t.training_set.pos_unique_indices) < self.max_occ:
                t.training_set.pos_unique_indices = self.generate_ind(t.training_set.pos_unique_indices, self.max_occ)
            pos_unique_indices = self.expand_indices_augmentation(t.training_set.pos_unique_indices)
            #pos_unique_indices_boxes = t.training_set.pos_unique_indices
            self.scores = torch.cat((self.scores, torch.ones(len(pos_unique_indices)).to(device) * (i+1)))
            #self.boxes = torch.cat((self.boxes, t.training_set.boxes[pos_unique_indices_boxes]))
            self.features = torch.cat((self.features, t.training_set.features[pos_unique_indices]))

        if val:
            val_set = self.get_val_set(val_idx, val_others_idx, inactive_tracks, others_dataset)
            return self, val_set

        return self, []


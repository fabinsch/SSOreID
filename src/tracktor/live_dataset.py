import torch
from torch import randperm
from torch.utils.data import Subset, ConcatDataset
from collections import defaultdict
from torchvision.ops.boxes import box_iou
import itertools
from operator import itemgetter


import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id, keep_frames, data_augmentation):
        self.id = id
        self.features = torch.tensor([]).to(device)
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.area = torch.tensor([]).to(device)
        self.keep_frames = keep_frames
        self.num_frames_keep = 0
        self.num_frames = 0
        self.pos_unique_indices = None
        self.data_augmentation = data_augmentation
        self.frame = torch.tensor([]).to(device)

    def append_samples(self, training_set_dict, frame=0, area=0):
        frame += 1 # because self.im_index is increased at the end of step
        self.num_frames += 1
        self.num_frames_keep += 1
        self.features = torch.cat((self.features, training_set_dict['features']))
        self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        self.scores = torch.cat((self.scores, torch.tensor([0]).float().to(device)))
        self.frame = torch.cat((self.frame, (torch.ones(1)*frame).to(device)))
        self.area = torch.cat((self.area, (torch.ones(1) * area).to(device)))
        if self.num_frames > self.keep_frames:
            self.remove_samples()
            self.num_frames_keep = self.keep_frames

    def remove_samples(self):
        self.boxes = self.boxes[1+self.data_augmentation:, :]
        self.scores = self.scores[1:]
        self.features = self.features[1+self.data_augmentation:, :, :, :]
        self.frame = self.frame[1:]
        self.area = self.area[1:]

    def post_process(self):
        self.pos_unique_indices = list(range(self.num_frames_keep))


    def __len__(self):
        return self.scores.size()[0]

    def __getitem__(self, idx):
        intervall = self.data_augmentation + 1  # has to be jumped to get corresponding features
        return {'features': self.features[idx*intervall:(idx*intervall+intervall), :, :, :], 'scores': self.scores[idx]*torch.ones(self.data_augmentation+1), 'frame': self.frame}


class InactiveDataset(torch.utils.data.Dataset):
    def __init__(self, data_augmentation=0, others_db=None, others_class=False, im_index=0, ids_in_others=0, val_set_random_from_middle=False, exclude_from_others=[], results=None):
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.features = torch.tensor([]).to(device)
        self.max_occ = 0
        self.min_occ = 0
        #self.killed_this_step = killed_this_step
        self.data_augmentation = data_augmentation
        self.others_db = others_db
        self.others_class = others_class
        self.im_index = im_index
        self.ids_in_others = ids_in_others
        self.val_set_random_from_middle = val_set_random_from_middle
        self.exclude_from_others = exclude_from_others
        #self.results = results
        self.frame = torch.tensor([]).to(device)

    def __len__(self):
        return self.scores.size()[0]

    def __getitem__(self, idx):
        #return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}
        return {'features': self.features[idx, :, :, :], 'scores': self.scores[idx], 'frame_id': self.frame[idx]}

    def balance(self, i, occ):
        if type(i) is list:
            if len(i) > 0:
                diff = occ - len(i)
                for j in range(diff):
                    i.append(random.choice(i))
                if self.im_index==207:
                    print("DEBUG balanced: {}".format(i))
                return i
            else:
                return []
        else:
            i, fID = i
            # balance the others class (already features)
            diff = occ - i.shape[0]
            duplicate = torch.LongTensor(diff).random_(0,i.shape[0])
            i = torch.cat((i, i[duplicate]))
            fID = torch.cat((fID, fID[duplicate]))
            return i, fID

    def get_others(self, inactive_tracks, val=False):
        val_others_features = torch.tensor([]).to(device)
        val_others_frames_id = torch.tensor([]).to(device)
        train_others_features = torch.tensor([]).to(device)
        train_others_frames_id = torch.tensor([]).to(device)

        inactive_ids = [t.id for t in inactive_tracks]
        #sorted_others_db = sorted(self.others_db.items(), key=itemgetter(1), reverse=True)
        #sorted_others_db_k = [t[0] for t in sorted_others_db]
        sorted_others_db_k = [t[0] for t in sorted(self.others_db.items(), key=itemgetter(1), reverse=True)]
        sorted_others_db_k = [k for k in sorted_others_db_k if k not in inactive_ids]
        others_db_k = list(self.others_db.keys())
        others_db_k = [k for k in others_db_k if k not in inactive_ids]
        #others_db_k = [k for k in others_db_k if k not in self.exclude_from_others]
        #print("excluding: {}".format(self.exclude_from_others))
        num_tracks = len(others_db_k)
        ids = self.ids_in_others
        if num_tracks >= 2:

            #num_frames = [t[0].shape[0] for t in list(self.others_db.values())]
            #num_frames_pp = [(t, num_frames[t]) for t in others_db_k]
            val_idx_others = sorted_others_db_k[0::2]
            train_idx_others = sorted_others_db_k[1::2]

            num_frames = [len(self.others_db[t]) for t in sorted_others_db_k]
            val_num_others = num_frames[0::2]
            train_num_others = num_frames[1::2]

            if val:
                if num_tracks >= 4:  # enough sample to provide at least two IDs in val and train others

                    print('\nIDs for val set others: {}'.format(val_idx_others))
                    print('IDs for train set others: {}'.format(train_idx_others))
                    print('IDs of inactive tracks: {}'.format(inactive_ids))

                    c = 0
                    for i, idx in enumerate(itertools.cycle(val_idx_others)):
                        i = i - len(val_idx_others)*c
                        if i >= len(val_idx_others):
                            c += 1
                            i = 0

                        if val_num_others[i]>0:
                            val_others_features = torch.cat((val_others_features, self.others_db[idx][c][1].unsqueeze(0)))
                            frames_id = torch.cat((self.others_db[idx][c][2].unsqueeze(0), (torch.ones(1)*idx).to(device)))
                            val_others_frames_id = torch.cat((val_others_frames_id, frames_id.unsqueeze(0)))
                            val_num_others[i] -= 1
                        if val_others_features.shape[0]>=(int(self.min_occ*0.2)*(self.data_augmentation+1)) or sum(val_num_others)==0:
                            break

                    c = 0
                    for i, idx in enumerate(itertools.cycle(train_idx_others)):
                        i = i - len(train_idx_others)*c
                        if i >= len(train_idx_others):
                            c += 1
                            i = 0
                        if train_num_others[i] > 0:
                            train_others_features = torch.cat((train_others_features, self.others_db[idx][c][1].unsqueeze(0)))
                            frames_id = torch.cat((self.others_db[idx][c][2].unsqueeze(0),
                                                   (torch.ones(1)*idx).to(device)))
                            train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))
                            train_num_others[i] -= 1
                        if train_others_features.shape[0]>=(self.max_occ*(self.data_augmentation+1)) or sum(train_num_others)==0:
                            break

                    print('There are {} train and {} val samples for others class'.format(train_others_features.shape[0], val_others_features.shape[0]))
                    return train_others_features, val_others_features, train_others_frames_id, val_others_frames_id

                else:  # just build train others because val others not divers
                    #train_others = num_frames_pp
                    #train_others = sorted(train_others, key=itemgetter(0), reverse=True)

                    # if ids > 0:
                    #     if len(train_others) > ids:
                    #         print("take just {} instead of {} different IDs".format(ids, len(train_others)))
                    #         train_others = train_others[:ids]
                    #
                    # train_idx_others, train_num_others = zip(*train_others)
                    # print('IDs for train set others: {}'.format(train_idx_others))
                    # train_num_others = list(train_num_others)

                    train_idx_others = sorted_others_db_k
                    train_num_others = [len(self.others_db[t]) for t in sorted_others_db_k]

                    c = 0
                    for i, idx in enumerate(itertools.cycle(train_idx_others)):
                        i = i - len(train_idx_others) * c
                        if i >= len(train_idx_others):
                            c += 1
                            i = 0
                        if train_num_others[i] > 0:
                            train_others_features = torch.cat(
                                (train_others_features, self.others_db[idx][c][1].unsqueeze(0)))
                            frames_id = torch.cat((self.others_db[idx][c][2].unsqueeze(0),
                                                   (torch.ones(1)*idx).to(device)))
                            train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))

                            train_num_others[i] -= 1
                        if train_others_features.shape[0] >= (self.max_occ*(self.data_augmentation+1)) or sum(train_num_others) == 0:
                            break
                    return train_others_features, val_others_features, train_others_frames_id, val_others_frames_id

            #### no validation set
            else:
                # train_others = num_frames_pp
                # train_others = sorted(train_others, key=itemgetter(0), reverse=True)
                # if ids > 0:
                #     if len(train_others) > ids:
                #         print("take just {} instead of {} different IDs".format(ids, len(train_others)))
                #         train_others = train_others[:ids]
                # train_idx_others, train_num_others = zip(*train_others)
                # train_num_others = list(train_num_others)

                train_idx_others = sorted_others_db_k
                train_num_others = [len(self.others_db[t]) for t in sorted_others_db_k]

                c = 0
                for i, idx in enumerate(itertools.cycle(train_idx_others)):
                    i = i - len(train_idx_others) * c
                    if i >= len(train_idx_others):
                        c += 1
                        i = 0
                    if train_num_others[i] > 0:
                        train_others_features = torch.cat(
                            (train_others_features, self.others_db[idx][c][1].unsqueeze(0)))
                        frames_id = torch.cat((self.others_db[idx][c][2].unsqueeze(0),
                                               (torch.ones(1) * idx).to(device)))
                        train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))
                        train_num_others[i] -= 1
                    if train_others_features.shape[0] >= (self.max_occ*(self.data_augmentation+1)) or sum(train_num_others) == 0:
                        break
                return train_others_features, val_others_features, train_others_frames_id, val_others_frames_id

        else:
            print('\n no others dataset. num tracks: {}'.format(num_tracks))
            return torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)

    def get_val_idx(self, occ, inactive_tracks, split, val_set_random):
        """generates indices for validation set und removes them from training set"""
        val_idx = []
        exclude_for_val = []
        just_one_sample_for_val = []
        org_min_occ = min(occ) if len(occ) > 0 else 0
        while True:
            min_occ = min(occ) if len(occ) > 0 else 0
            if min_occ >= 5 or min_occ == 0:  # >=5 means at least 1 val sample
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
                    print("0 validation sample for {}".format(idx_min))
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
                # else:
                #     if num_val > 1 and self.val_set_random_from_middle:
                #         # avoid that samples are taken from borders , but take random from middle
                #         pos_ind = t.training_set.pos_unique_indices[num_val:len(t.training_set.pos_unique_indices)-num_val]
                #         ri = random.choice(pos_ind)
                #         idx.append(ri)
                #         t.training_set.pos_unique_indices.remove(ri)
                #     else:
                #         # take samples from middle of scene to avoid taking the last occluded ones
                #         pos_ind = t.training_set.pos_unique_indices
                #         idx_val = (int(len(pos_ind) / 2) + int(num_val * 0.5))
                #         idx.append(pos_ind.pop(idx_val))

            val_idx.append(idx)

        num_val = int(self.min_occ * split)
        return val_idx, num_val

    def get_val_set(self, val_idx, inactive_tracks, val_others_features, val_others_fId):
        val_set = InactiveDataset()

        if val_others_features.shape[0] > 0:
            val_set.scores = torch.zeros(val_others_features.shape[0]).to(device)
            val_set.features = torch.cat((val_set.features, val_others_features))
            val_set.frame = torch.cat((val_set.frame, val_others_fId))
        # else:
        #     print('no validation set for others class')

        for i, idxs in enumerate(val_idx):
            c = i+1 if self.others_class else i
            # if len(idxs) < int(val_others_features.shape[0]/(self.data_augmentation+1)) and len(idxs) > 0:
            #     idxs = self.balance(idxs, int(val_others_features.shape[0]/(self.data_augmentation+1)))
            idx_features = self.expand_indices_augmentation(idxs)
            t = inactive_tracks[i]
            val_set.scores = torch.cat((val_set.scores, torch.ones(len(idx_features)).to(device) * (c)))
            #val_set.boxes = torch.cat((val_set.boxes, t.training_set.boxes[idxs]))
            val_set.features = torch.cat((val_set.features, t.training_set.features[idx_features]))
            if len(idxs)>0:
                val_set.frame = torch.cat((val_set.frame, torch.cat((t.training_set.frame[idxs].unsqueeze(1), torch.ones(len(idxs),1).to(device)*(t.id)), dim=1)))

        # if self.im_index == 117:
        #     id29 = torch.load('id29.pt')
        #     idx_29 = torch.arange(16,24).to(device)
        #     val_set.scores = torch.cat((val_set.scores, torch.ones(idx_29.shape).to(device) * (4)))
        #     val_set.features = torch.cat((val_set.features, id29[idx_29]))
        #     fId29 = torch.cat((idx_29.unsqueeze(1), (torch.ones_like(idx_29).to(device) * 29).unsqueeze(1)), dim=1)
        #     val_set.frame = torch.cat((val_set.frame, fId29.float()))
        #
        # if self.im_index == 117:
        #     id27 = torch.load('id27.pt')
        #     idx_27 = torch.arange(16, 24).to(device)
        #     val_set.scores = torch.cat((val_set.scores, torch.ones(idx_27.shape).to(device) * (5)))
        #     val_set.features = torch.cat((val_set.features, id27[idx_27]))
        #     fId27 = torch.cat((idx_27.unsqueeze(1), (torch.ones_like(idx_27).to(device) * 27).unsqueeze(1)), dim=1)
        #     val_set.frame = torch.cat((val_set.frame, fId27.float()))

        return val_set

    def expand_indices_augmentation(self, indices):
        idx = []
        inv = self.data_augmentation+1
        for i in indices:
            idx.extend(list(range(i*inv,i*inv+inv)))
        return idx


    def get_training_set(self, inactive_tracks, tracks, val, split, val_set_random, keep_frames):
        # if self.im_index==117:
        #    inactive_tracks = inactive_tracks[:-1]
        occ = [t.training_set.num_frames_keep for t in inactive_tracks]
        self.max_occ = max(occ) if len(occ) > 0 else 0

        val_idx = [[]]

        # get idx of validation samples
        if val:
            #val_idx, num_val = self.get_val_idx(occ, inactive_tracks, split, val_set_random)
            val_idx = [[]]
            num_val = 0
            #self.max_occ -= num_val

        self.min_occ = 40 # get 8 others samples for validation

        if self.others_class:
            # get others dataset with label 0
            train_others_features, val_others_features, train_other_fId, val_others_fId = self.get_others(inactive_tracks, val)
            if train_others_features.shape[0] < (self.max_occ*(self.data_augmentation+1)) and train_others_features.shape[0] > 0:
                train_others_features, train_other_fId = self.balance((train_others_features, train_other_fId), self.max_occ*(self.data_augmentation+1))
                if train_others_features.shape[0] == 0:
                    print('keine other tracks , nicht zu augmenten')
                #print('\nbalance because others is too less')
            self.scores = torch.zeros(train_others_features.shape[0]).to(device)
            self.features = torch.cat((self.features, train_others_features))
            self.frame = torch.cat((self.frame, train_other_fId))

        else:
            train_others_features = torch.tensor([]).to(device)
            val_others_features = torch.tensor([]).to(device)
            val_others_fId = torch.tensor([]).to(device)

        for i, t in enumerate(inactive_tracks):
            c = i+1 if self.others_class else i
            # balance dataset, same number of examples for each class
            #max_occ = max(self.max_occ, int(train_others_features.shape[0]/(self.data_augmentation+1)))
            if len(t.training_set.pos_unique_indices) < self.max_occ:
                t.training_set.pos_unique_indices = self.balance(t.training_set.pos_unique_indices, self.max_occ)
            pos_unique_indices = self.expand_indices_augmentation(t.training_set.pos_unique_indices)
            self.scores = torch.cat((self.scores, torch.ones(len(pos_unique_indices)).to(device) * (c)))
            #self.boxes = torch.cat((self.boxes, t.training_set.boxes[pos_unique_indices_boxes]))
            self.features = torch.cat((self.features, t.training_set.features[pos_unique_indices]))
            self.frame = torch.cat((self.frame, torch.cat((t.training_set.frame[pos_unique_indices].unsqueeze(1), torch.ones(len(pos_unique_indices),1).to(device)*(t.id)), dim=1)))

        # if self.im_index == 117:
        #     id29 = torch.load('id29.pt')
        #     idx_29 = torch.cat((torch.arange(16), torch.arange(24, 40))).to(device)
        #     self.scores = torch.cat((self.scores, torch.ones(idx_29.shape).to(device) * (4)))
        #     self.features = torch.cat((self.features, id29[idx_29]))
        #     fId29 = torch.cat((idx_29.unsqueeze(1), (torch.ones_like(idx_29).to(device)*29).unsqueeze(1)), dim=1)
        #     self.frame = torch.cat((self.frame, fId29.float()))
        #
        # if self.im_index == 117:
        #     id27 = torch.load('id27.pt')
        #     idx_27 = torch.cat((torch.arange(16), torch.arange(24, 40))).to(device)
        #     self.scores = torch.cat((self.scores, torch.ones(idx_27.shape).to(device) * (5)))
        #     self.features = torch.cat((self.features, id27[idx_27]))
        #     fId27 = torch.cat((idx_27.unsqueeze(1), (torch.ones_like(idx_27).to(device)*27).unsqueeze(1)), dim=1)
        #     self.frame = torch.cat((self.frame, fId27.float()))

        if val:
            val_set = self.get_val_set(val_idx, inactive_tracks, val_others_features, val_others_fId)
            return self, val_set

        return self, []


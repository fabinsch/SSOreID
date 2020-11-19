import torch
from torch import randperm
from torch.utils.data import Subset, ConcatDataset
from collections import defaultdict
from torchvision.ops.boxes import box_iou
import itertools
from operator import itemgetter
import logging
import time

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('main.live_dataset')
class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id, keep_frames, data_augmentation, flip_p):
        self.id = id
        #self.features = torch.tensor([]).to(device)
        self.features = torch.tensor([])
        #self.boxes = torch.tensor([]).to(device)
        self.boxes = torch.tensor([])
        #self.scores = torch.tensor([]).to(device)
        self.scores = torch.tensor([])
        #self.area = torch.tensor([]).to(device)
        self.area = torch.tensor([])
        self.keep_frames = keep_frames
        self.num_frames_keep = 0
        self.num_frames = 0
        self.pos_unique_indices = None
        self.data_augmentation = data_augmentation
        #self.frame = torch.tensor([]).to(device)
        self.frame = torch.tensor([])
        self.flip_p = flip_p

    def append_samples(self, training_set_dict, frame=0, area=0):
        frame += 1 # because self.im_index is increased at the end of step
        self.num_frames += 1
        self.num_frames_keep += 1
        self.features = torch.cat((self.features, training_set_dict['features']))
        self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        # self.scores = torch.cat((self.scores, torch.tensor([0]).float().to(device)))
        # self.frame = torch.cat((self.frame, (torch.ones(1)*frame).to(device)))
        # self.area = torch.cat((self.area, (torch.ones(1) * area).to(device)))
        self.scores = torch.cat((self.scores, torch.tensor([0]).float()))
        self.frame = torch.cat((self.frame, (torch.ones(1)*frame)))
        self.area = torch.cat((self.area, (torch.ones(1) * area)))
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
        ## while working with data augmentation
        # intervall = self.data_augmentation + 1  # has to be jumped to get corresponding features
        # return {'features': self.features[idx*intervall:(idx*intervall+intervall), :, :, :], 'scores': self.scores[idx]*torch.ones(self.data_augmentation+1), 'frame': self.frame}

        if self.flip_p>0.0:
            ## flip FM to augment
            features = self.features[idx, :, :, :].flip(-1)
            features = torch.cat((self.features[idx, :, :, :], features))
            scores = torch.cat((self.scores[idx], self.scores[idx]))
            frame_id = torch.cat((self.frame[idx], self.frame[idx]))

        else:
            features = self.features[idx, :, :, :]
            scores = self.scores[idx]
            frame_id =  self.frame[idx]

        return {'features': features.to(device), 'scores': scores.to(device) ,
                    'frame_id': frame_id.to(device)}


class InactiveDataset(torch.utils.data.Dataset):
    def __init__(self, data_augmentation=0, others_db=None, others_class=False, im_index=0,
                 ids_in_others=0, val_set_random_from_middle=False, exclude_from_others=[], results=None, flip_p=0.5,
                 fill_up=False, fill_up_to=10, flexible=False, upsampling=False, weightedLoss=False, samples_per_ID=1,
                 train_others=True, load_others=False):
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.features = torch.tensor([]).to(device)
        self.max_occ = 0
        self.min_occ = 0
        #self.killed_this_step = killed_this_step
        self.data_augmentation = data_augmentation
        # if type(others_db) is tuple:
        #     self.others_db = others_db[0]
        #     self.others_db_area = others_db[1]
        # else:
        #     self.others_db = others_db
        #     self.others_db_area = None
        if type(others_db) is tuple:
            self.others_db = others_db[0]
            #self.others_db_loaded = others_db[1][0]
            self.others_db_loaded = {}
            #self.others_db_loaded_id = others_db[1][1]
            self.others_db_area = None
        self.others_class = others_class
        self.im_index = im_index
        self.ids_in_others = ids_in_others
        self.val_set_random_from_middle = val_set_random_from_middle
        self.exclude_from_others = exclude_from_others
        #self.results = results
        self.frame = torch.tensor([]).to(device)
        self.loaded_others = load_others  # take number of sampes per ID in others, otherwise take all
        self.flip_p = flip_p
        self.fill_up = fill_up
        self.fill_up_to = fill_up_to
        self.flexible = flexible
        self.ratio = torch.tensor([]).to(device)
        self.upsampling = upsampling
        self.weightedLoss = weightedLoss
        self.samples_per_ID = samples_per_ID
        self.train_others = train_others

    def __len__(self):
        return self.scores.size()[0]

    def __getitem__(self, idx):
        #return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}
        #return {'features': self.features[idx, :, :, :], 'scores': self.scores[idx], 'frame_id': self.frame[idx]}
        return {'features': self.features[idx, :, :, :], 'scores': self.scores[idx]}

    def balance(self, i, occ):
        if type(i) is list:
            if len(i) > 0:
                diff = occ - len(i)
                for j in range(diff):
                    i.append(random.choice(i))
                # if self.im_index==207:
                #     print("DEBUG balanced: {}".format(i))
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

    def update_others(self, inactive_tracks):
        train_others_features = torch.tensor([]).to(device)
        train_others_frames_id = torch.tensor([]).to(device)

        inactive_ids = [t.id for t in inactive_tracks]
        sorted_others_db_k = [t[0] for t in self.others_db.items()]
        random.shuffle(sorted_others_db_k)
        sorted_others_db_k = [k for k in sorted_others_db_k if k not in inactive_ids]
        train_idx_others = sorted_others_db_k
        #train_num_others = [len(self.others_db[t]) for t in sorted_others_db_k]
        for i, idx in enumerate(train_idx_others):
            N = self.samples_per_ID
            s = torch.randint(len(self.others_db[idx]), (1, N))
            for j in range(s.shape[1]):
                train_others_features = torch.cat(
                    (train_others_features, self.others_db[idx][s[:, j]][1].unsqueeze(0).to(device)))
                frames_id = torch.cat((self.others_db[idx][s[:, j]][2].unsqueeze(0).to(device),
                                       (torch.ones(1) * idx).to(device)))
                train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))
        if self.flip_p > 0.0:
            ## flip FM to augment
            features = train_others_features.flip(-1)
            train_others_features = torch.cat((train_others_features, features))
            train_others_frames_id = torch.cat((train_others_frames_id, train_others_frames_id))

            self.features = self.features[self.ratio[0].long() * 2:, :, :, :]  # times 2 because of flip
        else:
            self.features = self.features[self.ratio[0].long():, :, :, :]  #

        #self.scores = torch.zeros(train_others_features.shape[0]).to(device)
        # first cut old others, than concat new ones

        self.features = torch.cat((train_others_features, self.features))
        #self.frame = torch.cat((self.frame, train_other_fId))
        return self

    def get_others(self, inactive_tracks, val=False):
        # FIRST idea get more others because of flip
        # if self.flip_p > 0.0:
        #     max_occ = 2*self.max_occ  # to balance with inactive tracks (each sample gets flipped feature map)
        # else:
        #     max_occ = self.max_occ

        # SECOND idea just flip others too, see below
        max_occ = self.max_occ

        val_others_features = torch.tensor([]).to(device)
        val_others_frames_id = torch.tensor([]).to(device)
        train_others_features = torch.tensor([]).to(device)
        train_others_frames_id = torch.tensor([]).to(device)
        fill_others_features = torch.tensor([]).to(device)
        fill_others_id = torch.tensor([]).to(device)
        #fill_others_frames_id = torch.tensor([]).to(device)

        # how many IDs from others

        if self.fill_up and len(self.others_db)>=self.fill_up_to: # make sure to have at least 4 IDs for others if 9 used for fil
            fill_up = self.fill_up_to-len(inactive_tracks) if self.fill_up_to-len(inactive_tracks)>0 else 0
            logger.debug('fill up {}'.format(fill_up))
        else:
            fill_up=0

        inactive_ids = [t.id for t in inactive_tracks]

        # sort others according to area
        # if self.others_db_area==None:
        #     sorted_others_db_k = [t[0] for t in sorted(self.others_db.items(), key=itemgetter(1), reverse=True)]
        # else:
        #     sorted_others_db_k = [t[0] for t in sorted(self.others_db_area.items(), key=itemgetter(1), reverse=True)]

        if self.flexible:  # test with flexible "fill-up"
            flex_num_fill = 0
            fill_IDs = list(self.others_db.keys())
            fill_IDs = [k for k in fill_IDs if k not in inactive_ids]
            #logger.debug("exclude from others {}".format(self.exclude_from_others))
            #fill_IDs = [k for k in fill_IDs if k not in self.exclude_from_others]
            num_frames = [len(self.others_db[t]) for t in fill_IDs]
            ids_start = len(inactive_tracks) + 1 if self.others_class else len(
                inactive_tracks)  # 0 is others, than num inactive, than fill up
            for i, f in enumerate(fill_IDs):
                if num_frames[i]>=max_occ:
                    idx = torch.randint(num_frames[i], (1,max_occ))
                    for j in range(idx.shape[1]):
                        fill_others_features = torch.cat((fill_others_features, self.others_db[f][idx[:,j]][1].unsqueeze(0).to(device)))
                        fill_others_id = torch.cat(
                            (fill_others_id, (ids_start + flex_num_fill) * torch.ones(1).to(device).unsqueeze(0)))
                    flex_num_fill += 1
            logger.info('flexible fill up with {} Ids having {} samples each'.format(flex_num_fill, max_occ))
            self.flex_num_fill = flex_num_fill
            if self.flip_p > 0.0:
                features = fill_others_features.flip(-1)
                fill_others_features = torch.cat((fill_others_features, features))
                fill_others_id = fill_others_id.repeat_interleave(2)
            return train_others_features, val_others_features, train_others_frames_id, val_others_frames_id, fill_others_features, fill_others_id


        sorted_others_db_k = [t[0] for t in self.others_db.items()]  # don't sort when loaded
        #random.shuffle(sorted_others_db_k)
        fill_IDs = sorted_others_db_k[:fill_up] # use these IDs to fill up to 10 nodes

        sorted_others_db_k = [k for k in sorted_others_db_k if k not in (inactive_ids or fill_IDs)]

        # comnbination of online and loaded
        sorted_others_db_k_loaded = list(self.others_db_loaded.keys())
        #sorted_others_db_k_loaded = self.others_db_loaded_id.squeeze(0)

        #others_db_k = list(self.others_db.keys())
        others_db_k = list(self.others_db.keys())
        others_db_k = [k for k in others_db_k if k not in (inactive_ids or fill_IDs)]
        #others_db_k = [k for k in others_db_k if k not in self.exclude_from_others]
        #print("excluding: {}".format(self.exclude_from_others))
        num_tracks = len(others_db_k)
        ids = self.ids_in_others
        if (num_tracks >= 2) or self.fill_up:

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
                        if train_others_features.shape[0]>=(max_occ*(self.data_augmentation+1)) or sum(train_num_others)==0:
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
                        if train_others_features.shape[0] >= (max_occ*(self.data_augmentation+1)) or sum(train_num_others) == 0:
                            break

                    # flip others too
                    # if self.flip_p > 0.0:
                    #     ## flip FM to augment
                    #     features = train_others_features.flip(-1)
                    #     train_others_features = torch.cat((train_others_features, features))
                    #     features = val_others_features.flip(-1)
                    #     val_others_features = torch.cat((val_others_features, features))
                    #     train_others_frames_id = torch.cat((train_others_frames_id, train_others_frames_id))
                    #     val_others_frames_id = torch.cat((val_others_frames_id, val_others_frames_id))


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
                ids_start = len(inactive_tracks)+1 if self.others_class else len(inactive_tracks) # 0 is others, than num inactive, than fill up
                for i, f in enumerate(fill_IDs):
                    frames = len(self.others_db[f])
                    idx = torch.randint(frames, (1,max_occ))
                    for j in range(idx.shape[1]):
                        fill_others_features = torch.cat((fill_others_features, self.others_db[f][idx[:,j]][1].unsqueeze(0).to(device)))
                        fill_others_id = torch.cat((fill_others_id, (ids_start+i)*torch.ones(1).to(device).unsqueeze(0)))

                if self.others_class:
                    train_idx_others = sorted_others_db_k
                    train_num_others = [len(self.others_db[t]) for t in sorted_others_db_k]

                    # if db others is sorted according to area and also each ID has to highest area at idx 0
                    # c = 0
                    # for i, idx in enumerate(itertools.cycle(train_idx_others)):
                    #     i = i - len(train_idx_others) * c
                    #     if i >= len(train_idx_others):
                    #         c += 1
                    #         i = 0
                    #     if train_num_others[i] > 0:
                    #         train_others_features = torch.cat(
                    #             (train_others_features, self.others_db[idx][c][1].unsqueeze(0).to(device)))
                    #         frames_id = torch.cat((self.others_db[idx][c][2].unsqueeze(0).to(device),
                    #                                (torch.ones(1) * idx).to(device)))
                    #         train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))
                    #         train_num_others[i] -= 1
                    #     if train_others_features.shape[0] >= (max_occ*(self.data_augmentation+1)) or sum(train_num_others) == 0:
                    #         break


                    for i, idx in enumerate(train_idx_others):
                        # random sample from others DBs according to max occurance of inactive
                        # TODO remove False
                        #if not self.upsampling and not self.weightedLoss and False:
                        if not self.upsampling and not self.weightedLoss and True:
                            j = random.randint(0, len(self.others_db[idx])-1)
                            if train_num_others[i] > 0:
                                train_others_features = torch.cat(
                                    (train_others_features, self.others_db[idx][j][1].unsqueeze(0).to(device)))
                                frames_id = torch.cat((self.others_db[idx][j][2].unsqueeze(0).to(device),
                                                       (torch.ones(1) * idx).to(device)))
                                train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))
                                train_num_others[i] -= 1
                            if train_others_features.shape[0] >= (max_occ * (self.data_augmentation + 1)) or sum(
                                    train_num_others) == 0:
                                break
                        elif self.loaded_others:
                            # take N samples of all IDs
                            N = self.samples_per_ID
                            if len(self.others_db[idx]) >= N:
                                s = random.sample(range(len(self.others_db[idx])), N)
                            else:
                                s = torch.randint(len(self.others_db[idx]), (1, N)).squeeze()
                            for j in range(len(s)):
                                train_others_features = torch.cat(
                                    (train_others_features, self.others_db[idx][s[j]][1].unsqueeze(0).to(device)))
                                # frames_id = torch.cat((self.others_db[idx][s[j]][2].unsqueeze(0).to(device),
                                #                        (torch.ones(1) * idx).to(device)))
                                # train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))

                        else:
                            # take all samples of all IDs
                           for j in range(len(self.others_db[idx])):
                               train_others_features = torch.cat(
                                   (train_others_features, self.others_db[idx][j][1].unsqueeze(0).to(device)))
                               # frames_id = torch.cat((self.others_db[idx][j][2].unsqueeze(0).to(device),
                               #                        (torch.ones(1) * idx).to(device)))
                               # train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))

                    # combination online and loaded
                    for i, idx in enumerate(sorted_others_db_k_loaded):
                        # take N samples of all IDs
                        N = self.samples_per_ID
                        if len(self.others_db_loaded[idx]) >= N:
                            s = random.sample(range(len(self.others_db_loaded[idx])), N)
                        else:
                            s = torch.randint(len(self.others_db_loaded[idx]), (1, N)).squeeze()
                        #for j in range(len(s)):

                        train_others_features = torch.cat((train_others_features, self.others_db_loaded[idx][s].to(device)))
                            #frames_id = torch.cat((self.others_db_loaded[idx][s[j]][2].unsqueeze(0).to(device),
                            # frames_id = torch.cat((torch.zeros([]).unsqueeze(0).to(device),
                            #                        (torch.ones(1) * idx).to(device)))
                            # train_others_frames_id = torch.cat((train_others_frames_id, frames_id.unsqueeze(0)))

                    if self.weightedLoss:
                            #self.ratio = max_occ/train_others_features.shape[0]
                            self.ratio = torch.cat((self.ratio, torch.ones(1).to(device)*train_others_features.shape[0]))

                    # flip others too
                #if self.flip_p > 0.0:
                    ## flip FM to augment
                    #features = train_others_features.flip(-1)
                    #train_others_features = torch.cat((train_others_features, features))
                    #features = val_others_features.flip(-1)
                    #val_others_features = torch.cat((val_others_features, features))
                    #train_others_frames_id = torch.cat((train_others_frames_id, train_others_frames_id))
                    #val_others_frames_id = torch.cat((val_others_frames_id, val_others_frames_id))

                    #features = fill_others_features.flip(-1)
                    #fill_others_features = torch.cat((fill_others_features, features))
                    #fill_others_id = fill_others_id.repeat_interleave(2)
                return train_others_features, val_others_features, train_others_frames_id, val_others_frames_id, fill_others_features, fill_others_id

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
            r = t.training_set[idx_features]
            val_set.scores = torch.cat((val_set.scores, torch.ones(len(r['scores'])).to(device) * (c)))
            #val_set.boxes = torch.cat((val_set.boxes, t.training_set.boxes[idxs]))
            val_set.features = torch.cat((val_set.features, r['features']))
            if len(idxs)>0:
                val_set.frame = torch.cat((val_set.frame, torch.cat((r['frame_id'].unsqueeze(1),
                                                                     torch.ones(len(r['scores']),1).to(device)*(t.id)), dim=1)))

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
            #val_idx = [[]]
            #num_val = 0
            val_idx, num_val = self.get_val_idx(occ, inactive_tracks, split, val_set_random)
            self.max_occ -= num_val

        #self.min_occ = 40 # get 8 others samples for validation

        # get an others class if there is just 1 ID, otherwise training complicated, output node can stay just 1 but
        # at least i have samples for 0 and 1 not just 0
        #if self.others_class or len(inactive_tracks) == 1: not necessary anymore with fillup
        if self.others_class:
            get_others = True
        else:
            get_others = False

        if (get_others or self.fill_up or self.flexible) and self.train_others:
            # get others dataset with label 0
            start_time = time.time()
            train_others_features, val_others_features, train_other_fId, val_others_fId, fill_features, fill_id = self.get_others(inactive_tracks, val)
            logger.info("\n--- %s seconds --- for loading others" % (time.time() - start_time))
            if train_others_features.shape[0] < (self.max_occ*(self.data_augmentation+1)) and train_others_features.shape[0] > 0:
                train_others_features, train_other_fId = self.balance((train_others_features, train_other_fId), self.max_occ*(self.data_augmentation+1))
                if train_others_features.shape[0] == 0:
                    print('keine other tracks , nicht zu augmenten')
                #print('\nbalance because others is too less')
            self.scores = torch.zeros(train_others_features.shape[0]).to(device)
            self.features = torch.cat((self.features, train_others_features))
            self.frame = torch.cat((self.frame, train_other_fId))

            self.scores = torch.cat((self.scores, fill_id))
            self.features = torch.cat((self.features, fill_features))
            self.frame = torch.cat((self.frame, -1*torch.ones((fill_id.shape[0],2)).to(device)))

        else:
            train_others_features = torch.tensor([]).to(device)
            val_others_features = torch.tensor([]).to(device)
            val_others_fId = torch.tensor([]).to(device)
            self.ratio = torch.cat((self.ratio, torch.ones(1).to(device)))

        for i, t in enumerate(inactive_tracks):
            # if len(fill_id)>0:
            #     c = int((fill_id[-1]+1).item())
            # else:
            #     c = i+1 if get_others else i
            c = i+1 if get_others else i
            # balance dataset, same number of examples for each class
            #max_occ = max(self.max_occ, int(train_others_features.shape[0]/(self.data_augmentation+1)))

            # deactivated 09/10 because try balancing in loss for each inactive
            if not self.weightedLoss:
                if len(t.training_set.pos_unique_indices) < self.max_occ:
                    t.training_set.pos_unique_indices = self.balance(t.training_set.pos_unique_indices, self.max_occ)
            #self.ratio = torch.cat((self.ratio, torch.ones(1).to(device)*len(t.training_set.pos_unique_indices)))
            #pos_unique_indices = self.expand_indices_augmentation(t.training_set.pos_unique_indices)  # if data augmentation is used
            pos_unique_indices = t.training_set.pos_unique_indices  # if no data augmentation is used
            r = t.training_set[pos_unique_indices]
            if self.weightedLoss:
                self.ratio = torch.cat((self.ratio, torch.ones(1).to(device) * r['scores'].shape[0]))
            self.scores = torch.cat((self.scores, torch.ones(len(r['scores'])).to(device) * (c)))
            self.features = torch.cat((self.features, r['features']))
            self.frame = torch.cat((self.frame, torch.cat((r['frame_id'].unsqueeze(1),
                                                           torch.ones(len(r['scores']), 1).to(device) * (t.id)),
                                                          dim=1)))

            # self.scores = torch.cat((self.scores, torch.ones(len(pos_unique_indices)).to(device) * (c)))
            # #self.boxes = torch.cat((self.boxes, t.training_set.boxes[pos_unique_indices_boxes]))
            # self.features = torch.cat((self.features, t.training_set.features[pos_unique_indices]))
            # self.frame = torch.cat((self.frame, torch.cat((t.training_set.frame[pos_unique_indices].unsqueeze(1), torch.ones(len(pos_unique_indices),1).to(device)*(t.id)), dim=1)))

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


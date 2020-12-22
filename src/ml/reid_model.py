from sacred import Experiment
import sacred
import os.path as osp
import os
import numpy as np
import yaml
import time
import functools, operator

import torch
import torch.nn
from torch.autograd import grad
from torch.nn import functional as F

from tracktor.config import get_output_dir, get_tb_dir
import random
from ml.utils import load_dataset, get_ML_settings, get_plotter, save_checkpoint, sample_task

import learn2learn as l2l
from learn2learn.utils import clone_module, clone_parameters
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class reID_Model(torch.nn.Module):
    def __init__(self, head, predictor, n_list, lr=1e-3):
        super(reID_Model, self).__init__()
        self.head = head
        for n in n_list:
            n += 1 # using others neuron
            self.add_module(f"last_{n}", torch.nn.Linear(1024, n).to(device))

        # deactive to try without templates
        if lr > 0:
            lrs = [torch.ones_like(p) * lr for p in self.parameters()]
            lrs = [torch.normal(mean=lr, std=1e-4) for lr in lrs]
            lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
        self.lrs = lrs  # first 4 entries belonging to head, than 2 for each last N module first weight than bias

        self.num_output = n_list


    def forward(self, x, nways):
        feat = self.head(x)
        last_n = f"last_{nways + 1}"  # using others neuron
        #last_n = f"last_{nways}"

        # because of variable last name
        for name, layer in self.named_modules():
            if name == last_n:
                x = layer(feat)
        return x

    def remove_last(self, state_dict, num_output):
        r = dict(state_dict)
        for i, n in enumerate(num_output):
            key_weight = f"module.last_{n + 1}.weight"
            # key_weight = f"module.last_{n}.weight"
            key_bias = f"module.last_{n + 1}.bias"
            # key_bias = f"module.last_{n}.bias"
            key_lr_weight = f"module.lrs.{4 + (i * 2)}"
            key_lr_bias = f"module.lrs.{5 + (i * 2)}"
            del r[key_weight]
            del r[key_bias]
            del r[key_lr_weight]
            del r[key_lr_bias]
        return r

    def forward_pass_for_classifier_training(self, learner, features, labels, nways, return_scores=False, weights=[]):
        # occ = torch.sqrt(occ)
        # occ = torch.ones(nways + 1).to(device)
        # num_others = len(features[1])
        if type(features) is tuple:  # if val set includes others
            # first task, than others own seq
            tasks_others_own = torch.cat((features[0], features[1]))
            if len(features[2]) > 1:  # if others from all seq used in val set
                # third others from all others
                class_logits2 = learner(tasks_others_own, nways)
                class_logits = learner(features[2], nways)
                class_logits = torch.cat((class_logits2, class_logits))
            else:
                class_logits = learner(tasks_others_own, nways)

        else:  # if no others involved
            class_logits = learner(features, nways)

        if return_scores:
            pred_scores = F.softmax(class_logits, -1)
            # w = torch.ones(int(torch.max(labels).item()) + 1).to(device)
            # occ = torch.ones(nways + 1).to(device)
            # w = w * (1 / occ)
            # do not train on others, just eval
            # if occ[0] < 0:
            #    w[0] = 0

            # print(f"occ : {occ}")
            # print(f"weights : {w}")
            # print(f"weight before norm {w}")
            # exit()
            # w /= w.sum()
            # print(f"weight after norm {w}")
            # loss = F.cross_entropy(class_logits, labels.long(), weight=w)
            if len(weights) < 1:
                loss = F.cross_entropy(class_logits, labels.long())
            elif len(weights) == (nways + 1):  # class ratio
                loss = F.cross_entropy(class_logits, labels.long(), weight=weights)
            else:
                loss = F.cross_entropy(class_logits, labels.long(), reduction='none')
                loss = (loss * weights / weights.sum()).sum()  # needs to be sum at the end not mean!
            # loss = (loss * weights).mean()

            # loss2 = F.cross_entropy(class_logits[:15], labels[:15].long())  # just samples for task
            # loss3 = F.cross_entropy(class_logits[15:], labels[15:].long())  # just samples for class 0

            # loss_individual_zero = F.cross_entropy(class_logits[30].unsqueeze(0), labels[30].unsqueeze(0).long())
            # loss_individual_task = F.cross_entropy(class_logits[10].unsqueeze(0), labels[10].unsqueeze(0).long())
            # loss4 = F.cross_entropy(class_logits, labels.long())
            # print(loss3.item())
            # print(loss2.item())
            # print(loss.item())
            # print(loss==loss2)
            return pred_scores.detach(), loss

    def accuracy_noOthers(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        valid_accuracy_without_zero = (predictions == targets).sum().float() / targets.size(0)
        return (valid_accuracy_without_zero, torch.zeros(1), torch.zeros(1), torch.zeros(1))

    def accuracy(self, predictions, targets, iter=-1, train=False, seq=-1, num_others_own=-1):
        # in target there are first the labels of the tasks (1, N) N*K*2 times, afterwards the labels of others own (0)
        # num_others_own times, than labels for all others (0)
        predictions = predictions.argmax(dim=1).view(targets.shape)
        compare = (predictions == targets)

        non_zero_target = (targets != 0)
        num_non_zero_targets = non_zero_target.sum()

        zero_target = (targets == 0)
        num_zero_targets = zero_target.sum()

        correct_others_own = compare[num_non_zero_targets:(num_non_zero_targets + num_others_own)].sum()
        valid_accuracy_others_own = correct_others_own.float() / num_others_own
        if iter % 500 == 0 and False:
            name = 'train ' + seq if train else 'val'

            num_correct_predictions_zero_targets = compare[targets == 0].sum()
            zero_target_missed = num_zero_targets - num_correct_predictions_zero_targets

            num_correct_predictions_non_zero_targets = compare[targets != 0].sum()
            non_zero_target_missed = num_non_zero_targets - num_correct_predictions_non_zero_targets

            print(f"{name:<20} {non_zero_target_missed}/{num_non_zero_targets} persons missed, "
                  f"{zero_target_missed}/{num_zero_targets} others missed, "
                  f"{num_others_own - correct_others_own}/{num_others_own} others OWN sequence missed")

        valid_accuracy_with_zero = compare.sum().float() / targets.size(0)
        valid_accuracy_without_zero = compare[non_zero_target].sum().float() / num_non_zero_targets
        valid_accuracy_just_zero = compare[zero_target].sum().float() / num_zero_targets

        return (
        valid_accuracy_without_zero, valid_accuracy_with_zero, valid_accuracy_just_zero, valid_accuracy_others_own)

    def fast_adapt_noOthers(self, batch, learner, adaptation_steps, shots, ways, train_task=True,
                            plotter=None, iteration=-1, task=-1, taskID=-1, reid=None, seq=-1):

        flip_p = reid['ML']['flip_p']

        data, labels = batch
        data, labels = data, labels.to(device)
        n = 1  # consider flip in indices
        if flip_p > 0.0:
            n = 2
            # do all flipping here, put flips at the end
            data = torch.cat((data, data.flip(-1)))
            labels = labels.repeat(2)

        # Separate data into adaptation/evaluation sets
        train_indices = np.zeros(data.size(0), dtype=bool)
        train_indices[np.arange(shots * ways * n) * 2] = True  # true false true false ...
        val_indices = torch.from_numpy(~train_indices)
        train_indices = torch.from_numpy(train_indices)

        train_data, train_labels = data[train_indices], labels[train_indices]
        val_data, val_labels = data[val_indices], labels[val_indices]

        # init last layer with the template weights
        learner.init_last(ways)
        train_accuracies = []
        for step in range(adaptation_steps):
            train_predictions, train_loss = forward_pass_for_classifier_training(learner, train_data, train_labels,
                                                                                 ways, return_scores=True)
            train_accuracy = accuracy_noOthers(train_predictions, train_labels)
            train_accuracies.append(train_accuracy)
            learner.adapt(train_loss)  # Takes a gradient step on the loss and updates the cloned parameters in place

        predictions, validation_loss = forward_pass_for_classifier_training(learner, val_data, val_labels, ways,
                                                                            return_scores=True)
        valid_accuracy = accuracy_noOthers(predictions, val_labels)

        return validation_loss, valid_accuracy, train_accuracies

    def fast_adapt(self, batch, learner, adaptation_steps, shots, ways, train_task=True,
                   plotter=None, iteration=-1, task=-1, taskID=-1, reid=None,
                   others_own=torch.zeros(1), samples_per_id_others_own=torch.zeros(1),
                   others=torch.zeros(1), seq=-1, train_others=False, set=None):

        # get others from own seq
        others_own, others_own_id = set[others_own]
        flip_p = reid['ML']['flip_p']

        data, labels = batch
        data, labels = data, labels.to(device) + 1  # because 0 is for others class, transfer data to gpu
        occ = torch.ones(ways + 1).to(device) * shots
        n = 1  # consider flip in indices
        if flip_p > 0.0:
            n = 2
            # do all flipping here, put flips at the end
            data = torch.cat((data, data.flip(-1)))
            labels = labels.repeat(2)
            occ *= 2

        # Separate data into adaptation/evalutation sets
        train_indices = np.zeros(data.size(0), dtype=bool)
        train_indices[np.arange(shots * ways * n) * 2] = True  # true false true false ...
        val_indices = torch.from_numpy(~train_indices)
        train_indices = torch.from_numpy(train_indices)

        info = (seq, ways, shots, iteration, train_task, taskID)

        train_data, train_labels = data[train_indices], labels[train_indices]
        val_data, val_labels = data[val_indices], labels[val_indices]

        # fill placeholder with template neuron
        learner.init_last(ways)

        # Adapt the model
        if reid['ML']['weightening'] == 1:  # weights for task 1/number of neurons * 1/samples per ID
            task_weights = torch.ones(len(train_labels)) / ((ways + 1) * occ[1])
        elif reid['ML']['weightening'] == 2:  # weights for task 1/samples per ID
            task_weights = torch.ones(len(train_labels)) / occ[1]
        elif reid['ML']['weightening'] == 3:  # no weights
            occ = torch.ones(ways + 1).to(device)

        # use 10 percent in inner loop
        if len(others_own) > 10:
            indices = random.sample(range(len(others_own)), int(len(others_own) * 0.2))
            train_indices = torch.tensor(indices[::2])
            val_indices = torch.tensor(indices[1::2])

            others_own_inner_train = others_own[train_indices]

            if reid['ML']['weightening'] in [1, 2]:
                # calculate weights for others, train and val the same 10 %
                others_own_inner_train_id = others_own_id.squeeze()[train_indices].numpy()
                counter = Counter(others_own_inner_train_id)
                others_own_inner_train_occ = [counter[id] for id in others_own_inner_train_id]
                num_ids_others = len(counter)
                if reid['ML']['weightening'] == 1:  # weight is 1/number of neurons * 1/number of IDs * 1/samples per ID
                    others_own_inner_val_weight = torch.ones(1) / (
                                (ways + 1) * num_ids_others * torch.tensor(others_own_inner_train_occ))
                elif reid['ML']['weightening'] == 2:
                    others_own_inner_val_weight = torch.ones(1) / (
                                num_ids_others * torch.tensor(others_own_inner_train_occ))
                weights_val = torch.cat((task_weights, others_own_inner_val_weight)).to(device)
                if train_others:
                    weights_train = weights_val
                else:
                    others_own_inner_train_weight = torch.zeros(len(others_own_inner_train_id))
                    weights_train = torch.cat((task_weights, others_own_inner_train_weight)).to(device)

            # do the same for others from all seq if applicable
            if len(others) > 1:
                ind = random.sample(range(len(others)), int(len(others) * 0.2))
                ind_train = torch.tensor(ind[::2])
                ind_others_val = torch.tensor(ind[1::2])
                others_inner_train = others[ind_train]
            else:
                others_inner_train = torch.zeros(1)

            ID_others = 0
            num_others = len(others_inner_train) + len(others_own_inner_train) if len(others) > 1 else len(
                others_own_inner_train)

            train_labels = torch.cat((train_labels, torch.ones(num_others).long().to(device) * ID_others))


        if reid['ML']['weightening'] in [0, 3]:
            weights_val = torch.ones(1).to(device) / occ
            weights_val[ID_others] = torch.ones(1).to(device) / num_others
            if not train_others:
                weights_train = torch.ones(1).to(device) / occ
                weights_train[ID_others] = 0
            else:
                weights_train = weights_val

        for step in range(adaptation_steps):

            train_predictions, train_loss = self.forward_pass_for_classifier_training(
                learner,(train_data, others_own_inner_train, others_inner_train),train_labels, ways,
                return_scores=True,weights=weights_train)

            # # plot inner loop
            # train_accuracy = self.accuracy(train_predictions, train_labels, iteration,
            #                           train_task, seq, others_own_inner_train.shape[0])
            # if train_task and ((iteration % 10 == 0 and iteration <= 100) or iteration % 100 == 0) and task == 0 and False:
            #     # if train_task and iteration % 100 == 0 and task == 0 and True:
            #     plotter.plot(epoch=step + 1, loss=train_loss, acc=train_accuracy, split_name='inner', info=info)

            learner.adapt(train_loss)  # Takes a gradient step on the loss and updates the cloned parameters in place


        if len(others) > 1 or len(others_own) > 1:  # meta learn others class with others in val set
            # get 10 percent of others own
            if len(others_own) > 10:
                # others_own = others_own[ind_val]
                others_own = others_own[train_indices]  # TODO if this is changed also weights have to be changed
                if len(others) > 1:
                    others = others[ind_others_val]

            val_labels = torch.cat((val_labels, torch.ones(num_others).long().to(device) * ID_others))


            predictions, validation_loss = self.forward_pass_for_classifier_training(
                learner, (val_data, others_own, others), val_labels, ways, return_scores=True, weights=weights_val)

            valid_accuracy = self.accuracy(predictions, val_labels, iteration, train_task, seq, others_own.shape[0])

        return validation_loss, valid_accuracy
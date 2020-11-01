from sacred import Experiment
import sacred
import os.path as osp
import os
import numpy as np
import yaml
import cv2
import sys
import time
import random
import functools, operator

import torch
import torch.nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from collections import namedtuple

from tracktor.config import get_output_dir, get_tb_dir
from tracktor.reid.solver import Solver
from tracktor.datasets.factory import Datasets
from tracktor.reid.resnet import resnet50
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.visualization import VisdomLinePlotter_ML
import random
from tracktor.utils import HDF5Dataset

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData
from learn2learn.utils import clone_module, clone_parameters
import h5py
import datetime

#from tracktor.transforms import FusedNWaysKShots

ex = Experiment()
ex.add_config('experiments/cfgs/ML_reid.yaml')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    #print('[{}] safe checkpoint {}'.format(state['epoch'], filename))


def meta_sgd_update(model, lrs=None, grads=None):
    """
    **Description**
    Performs a MetaSGD update on model using grads and lrs.
    The function re-routes the Python object, thus avoiding in-place
    operations.
    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.
    **Arguments**
    * **model** (Module) - The model to update.
    * **lrs** (list) - The meta-learned learning rates used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.
    **Example**
    ~~~python
    meta = l2l.algorithms.MetaSGD(Model(), lr=1.0)
    lrs = [th.ones_like(p) for p in meta.model.parameters()]
    model = meta.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    meta_sgd_update(model, lrs=lrs, grads)
    ~~~
    """
    #diff_params = [p for p in model.parameters() if p.requires_grad]
    if grads is not None and lrs is not None:
        if model.global_LR:
            lr = lrs
            for p, g in zip(model.parameters(), grads):
                p.grad = g
                p._lr = lr
        else:
            for p, lr, g in zip(model.parameters(), lrs, grads):
                p.grad = g
                p._lr = lr

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - p._lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = meta_sgd_update(model._modules[module_key])
    #model._apply(lambda x: x)
    return model


class MAML(l2l.algorithms.MAML):
    def __init__(self,
                 model,
                 lr,
                 first_order=False,
                 allow_unused=None,
                 allow_nograd=False):
        super(l2l.algorithms.MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused


    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)


    def init_last(self, ways):
        last_n = 'last_{}'.format(ways-1)
        repeated_bias = self.predictor.bias.clone().repeat(ways - 1)
        repeated_weights = self.predictor.weight.clone().repeat(ways - 1, 1)
        for name, layer in self.module.named_modules():
            if name == last_n:
                for param_key in layer._parameters:
                    p = layer._parameters[param_key]
                    #p.requires_grad = True
                    if param_key=='weight':
                        p.data = repeated_weights
                    elif param_key=='bias':
                        p.data = repeated_bias


class MetaSGD(l2l.algorithms.MetaSGD):
    def __init__(self, model, lr=1.0, first_order=False, allow_unused=None, allow_nograd=False, lrs=None, global_LR=False):
        super(l2l.algorithms.MetaSGD, self).__init__()
        self.module = model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.module.global_LR = global_LR
        # LR per layer
        # if lrs is None:
        #     lrs = [torch.ones(1).to(device) * lr for p in model.parameters()]
        #     lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])

        if global_LR:
            # one global LR
            if lrs is None:
                lrs = [torch.nn.Parameter(torch.ones(1).to(device) * lr)]
            self.lrs = lrs[0]
        else:
            if lrs is None:
                lrs = [torch.ones_like(p) * lr for p in model.parameters()]
                print('quick fix to exclude LRs for additional layer')
                lrs = lrs[:-2]  # quick fix to exclude LRs for additional layer
                lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
            self.lrs = lrs

        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Descritpion**
        Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
        learning rates.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MetaSGD(clone_module(self.module),
                       lrs=clone_parameters(self.lrs),
                       first_order=first_order,
                       allow_unused=allow_unused,
                       allow_nograd=allow_nograd,
                       global_LR=self.module.global_LR)

    def adapt(self, loss, first_order=None, allow_nograd=False):
        """
        **Descritpion**
        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
        if first_order is None:
            first_order = self.first_order

        second_order = not first_order
        diff_params = [p for p in self.module.parameters() if p.requires_grad]
        gradients = grad(loss,
                         diff_params,
                         retain_graph=second_order,
                         create_graph=second_order,
                         allow_unused=self.allow_unused)
        # gradients = []
        # grad_counter = 0
        #
        # # Handles gradients for non-differentiable parameters
        # for param in self.module.parameters():
        #     if param.requires_grad:
        #         gradient = grad_params[grad_counter]
        #         grad_counter += 1
        #     else:
        #         gradient = None
        #     gradients.append(gradient)

        self.module = meta_sgd_update(self.module, self.lrs, gradients)

    def init_last(self, ways, train=False):
        if self.module.global_LR:
            last_n = 'last_{}'.format(ways - 1)
            repeated_bias = self.predictor.bias.clone().repeat(ways - 1)
            repeated_weights = self.predictor.weight.clone().repeat(ways - 1, 1)
            for name, layer in self.module.named_modules():
                if name == last_n:
                    for param_key in layer._parameters:
                        p = layer._parameters[param_key]
                        # p.requires_grad = True
                        if param_key == 'weight':
                            p.data = repeated_weights
                        elif param_key == 'bias':
                            p.data = repeated_bias
                    break
        else:
            #last = ways-1 if train else ways
            last = ways  # always others neuron + 4 times duplicated template neuron
            last_n = 'last_{}'.format(last)
            additional = ways-1
            repeated_bias = self.predictor.bias.clone().repeat(additional)
            #repeated_bias_lr = self.lrs[5].clone().repeat(additional)
            repeated_bias_lr = self.lrs[7].clone().repeat(additional)
            repeated_weights = self.predictor.weight.clone().repeat(additional, 1)
            #repeated_weights_lr = self.lrs[4].clone().repeat(additional, 1)
            repeated_weights_lr = self.lrs[6].clone().repeat(additional, 1)

            repeated_weights = torch.cat((self.module.others_neuron_weight, repeated_weights))
            repeated_bias = torch.cat((self.module.others_neuron_bias, repeated_bias))
            repeated_weights_lr = torch.cat((self.lrs[0], repeated_weights_lr))
            repeated_bias_lr = torch.cat((self.lrs[1], repeated_bias_lr))
            for name, layer in self.module.named_modules():
                if name == last_n:
                    for param_key in layer._parameters:
                        p = layer._parameters[param_key]
                        #p.requires_grad = True
                        if param_key=='weight':
                            p.data = repeated_weights
                            self.lrs.append(repeated_weights_lr)
                        elif param_key=='bias':
                            p.data = repeated_bias
                            self.lrs.append(repeated_bias_lr)
                    # layer.bias.requires_grad = True
                    # layer.weight.requires_grad = True
                    # layer.bias.data = repeated_bias
                    # layer.weight.data = repeated_weights
                    break

    def last_layers_freeze(self):
        last_layers = ['last_{}'.format(n-1) for n in self.num_output]
        # for name, layer in self.module.named_modules():
        #     if name in last_layers:
        #         for param in layer.parameters():
        #             param.requires_grad = False
                #layer.bias.requires_grad = False
                #layer.weight.requires_grad = False

    # def init_last(self, ways):
    #     last_n = 'last_{}'.format(ways-1)
    #     repeated_bias = self.module.predictor.bias.repeat(ways - 1)
    #     repeated_weights = self.module.predictor.weight.clone().repeat(ways - 1, 1)
    #     for name, layer in self.module.named_modules():
    #         if name == last_n:
    #             layer.bias.data = repeated_bias
    #             layer.weight.data = repeated_weights
    #     #self.module.last.bias.data = torch.ones(4).to(device)
    #     # last = self.additional_layer[ways].to(device)
    #     # cm = clone_module(self.module)
    #     # cm.add_module('last', torch.nn.Linear(1024, ways-2).to(device))
    #     # self.module = cm
    #     #self.last.weight.requires_grad = False
    #     #self.last.bias.requires_grad = False
    #
    #     # weight = self.predictor.weight
    #     # weight_repeated = weight.repeat(ways, 1)
    #     # bias = self.predictor.bias
    #     # bias_repeated = bias.repeat(ways)
    #     #
    #     #repeated_bias = self.predictor.bias.clone().repeat(ways-1)
    #     #repeated_weights = self.predictor.weight.clone().repeat(ways-1, 1)
    #     #with torch.no_grad():
    #     #self.last.weight = torch.nn.Parameter(repeated_weights)
    #     #self.last.bias = torch.nn.Parameter(repeated_bias)
    #
    #     #self.last.weight.data.fill_(self.predictor.weight.repeat(ways-1, 1))
    #     #self.last.bias.data.fill_(self.predictor.bias.repeat(ways-1))
    #     return self

class ML_dataset(Dataset):
    def __init__(self, data, flip_p):
        # shift FM to GPU
        self.fm = data[1].to(device)
        #self.id = data[0].to(device)
        self.id = data[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.flip_p = flip_p
        self.flip_p = -1
    def __getitem__(self, item):
        # First way flip was implemented, either flipped or non flipped sample as output
        # if random.random() < self.flip_p:
        #     features = self.data[1][item].flip(-1)
        # else:
        #     features = self.data[1][item]
        # return (features, self.data[0][item].unsqueeze(0))

        # In reID network in tracktor we will always use non flipped and flipped version
        # do the same here in ML setting

        if self.flip_p>0.0:
            features = self.data[1][item].flip(-1).unsqueeze(0)
            features = torch.cat((self.data[1][item].unsqueeze(0), features))

            label = self.data[0][item] #.unsqueeze(0)
            #label = torch.cat((self.data[0][item].unsqueeze(0), label))
            return (features.to(device), label)
        else:
            return (self.fm[item], self.id[item].unsqueeze(0))

    def __len__(self):
        return len(self.id)


class reID_Model(torch.nn.Module):
    def __init__(self, head, predictor, n_list):
        super(reID_Model, self).__init__()
        self.head = head
        self.predictor = predictor.cls_score
        #self.last = torch.nn.Linear(1024, 4).to(device)
        # just get 1 template neuron as output and additional last X (to be filled with template)
        self.others_neuron_weight = torch.nn.Parameter(self.predictor.weight[1, :].unsqueeze(0))
        self.others_neuron_bias = torch.nn.Parameter(self.predictor.bias[1].unsqueeze(0))
        self.predictor.weight = torch.nn.Parameter(self.predictor.weight[0,:].unsqueeze(0))
        self.predictor.bias = torch.nn.Parameter(self.predictor.bias[0].unsqueeze(0))


        for n in n_list:
            #self.add_module('last_{}'.format(n - 1), torch.nn.Linear(1024, n - 1).to(device)) ## before having others neuron
            n += 1
            self.add_module('last_{}'.format(n - 1), torch.nn.Linear(1024, n - 1).to(device))
        self.num_output = n_list
        # self.additional_layer = {}
        # if len(n)>0:
        #     for i in n:
        #         if i>2:
        #             self.additional_layer[i] = torch.nn.Linear(1024, i - 2)
        #         else:
        #             self.additional_layer[i] = None
        # else:
        #     self.additional_layer = None

    def forward(self, x, nways, train):
        #nways += 0 if train else 1
        nways += 1

        feat = self.head(x)
        x = self.predictor(feat)
        # if self.additional_layer != None and nways>2:
        #     self.additional_layer[nways].to(device)
        #     add = self.additional_layer[nways](feat)
        #     x = torch.cat((x, add), dim=1)

        if nways>1:
            last_n = 'last_{}'.format(nways - 1)
            for name, layer in self.named_modules():
                if name == last_n:
                    add = layer(feat)
                    break
            x = torch.cat((add, x), dim=1)
        # else:

        return x
    # def init_last(self, ways):
    #     last_n = 'last_{}'.format(ways-1)
    #     repeated_bias = self.predictor.bias.repeat(ways - 1)
    #     repeated_weights = self.predictor.weight.clone().repeat(ways - 1, 1)
    #     for name, layer in self.named_modules():
    #         if name == last_n:
    #             layer.bias.requires_grad = True
    #             layer.weight.requires_grad = True
    #             layer.bias.data = repeated_bias
    #             layer.weight.data = repeated_weights

    def last_layers_freeze(self):
        last_layers = ['last_{}'.format(n-1) for n in self.num_output]
        for name, layer in self.named_modules():
            if name in last_layers:
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

def remove_last(state_dict, num_output):
    r = dict(state_dict)
    for n in num_output:
        n += 1
        key_weight = 'module.last_{}.weight'.format(n-1)
        key_bias = 'module.last_{}.bias'.format(n-1)
        del r[key_weight]
        del r[key_bias]
    return r


def forward_pass_for_classifier_training(learner, features, scores, nways, eval=False, return_scores=False, ratio=[], train=False):

    if eval:
        learner.eval()
    class_logits = learner(features, nways, train)

    if return_scores:
        pred_scores = F.softmax(class_logits, -1)
        if not train:
            w = torch.ones(int(torch.max(scores).item()) + 1).to(device)
            # w[0] = w[0]*ratio  # downweight 0 class
            # w[1:] = w[1:]*(1/ratio)  # upweight inactive
            w = w * (1 / ratio)
            loss = F.cross_entropy(class_logits, scores.long(), weight=w)
        else:
            loss = F.cross_entropy(class_logits, scores.long())
        if eval:
            learner.train()
        return pred_scores.detach(), loss

    if not train:
        w = torch.ones(int(torch.max(scores).item()) + 1).to(device)
        # w[0] = w[0]*ratio  # downweight 0 class
        # w[1:] = w[1:]*(1/ratio)  # upweight inactive
        w = w * (1 / ratio)
        w = torch.cat((torch.ones(1).to(device),w)) if train else w
        loss = F.cross_entropy(class_logits, scores.long(), weight=w)
    else:
        loss = F.cross_entropy(class_logits, scores.long())

    if eval:
        learner.train()
    else:
        return loss


def accuracy(predictions, targets, iter=-1, train=False):
    valid_accuracy_with_zero = torch.zeros(1)
    predictions = predictions.argmax(dim=1).view(targets.shape)
    #print(predictions)
    non_zero_target = (targets != 0)
    num_non_zero_targets = non_zero_target.sum()
    if iter%100==0 and not train:
        zero_target = (targets == 0)

        #zero_predict = (predictions == 0)
        compare = (predictions == targets)
        #zero_missed = (compare==False).sum()
        num_zero_targets = zero_target.sum()

        num_correct_predictions_zero_targets = compare[num_non_zero_targets:].sum()
        zero_target_missed = num_zero_targets-num_correct_predictions_zero_targets
        num_correct_predictions_non_zero_targets = compare[:num_non_zero_targets].sum()
        non_zero_target_missed = num_non_zero_targets-num_correct_predictions_non_zero_targets
        print('{}/{} persons missed, {}/{} others from own sequence missed'.format(non_zero_target_missed, num_non_zero_targets, zero_target_missed, num_zero_targets))

    valid_accuracy_without_zero = (predictions[:num_non_zero_targets] == targets[:num_non_zero_targets]).sum().float() / targets[:num_non_zero_targets].size(0)
    if not train:
        valid_accuracy_with_zero = (predictions == targets).sum().float() / targets.size(0)
        #valid_accuracy_just_zero = 0.0
        valid_accuracy_just_zero = (predictions[num_non_zero_targets:] == targets[num_non_zero_targets:]).sum().float() / targets[num_non_zero_targets:].size(0)
        return (valid_accuracy_without_zero, valid_accuracy_with_zero, valid_accuracy_just_zero)

    return valid_accuracy_without_zero

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.best = 1000
        self.best_it = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def update_best(self, val, it):
        save = 0
        if val < self.best:
            self.best = val
            self.best_it = it
            save = 1
        return save


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def fast_adapt(batch, learner, adaptation_steps, shots, ways,  device, train_task=True,
               plotter=None, iteration=-1, sequence=None, task=-1, taskID=-1, reid=None,
               init_last=True, others_own=None, others=torch.zeros(1)):
    #print('fast adapt')
    flip_p = reid['ML']['flip_p']
    valid_accuracy_before = torch.zeros(1)
    validation_error_before = torch.zeros(1)
    accuracy_others = torch.zeros(1)


    data, labels = batch
    data, labels = data, labels.to(device)+1  ## because 0 is for others class
    ratio = torch.ones(ways + 1).to(device) * shots
    n = 1 # consider flip in indices
    if flip_p > 0.0:
        n = 2
        # get flipped feature maps, in combination with getitem method
        #data = data.view(-1, 256, 7, 7)
        #labels = labels.repeat_interleave(2)
        # because of flip
        #adaptation_indices = adaptation_indices.repeat_interleave(2)
        #evaluation_indices = evaluation_indices.repeat_interleave(2)

        # do all flipping here, put flips at the end
        start_time = time.time()
        data = torch.cat((data, data.flip(-1)))
        labels = labels.repeat(2)


        ratio = torch.ones(ways + 1).to(device) * shots * 2
        #print("--- %s seconds --- [fa] flip " % (time.time() - start_time))

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways * n) * 2] = True  # true false true false ...
    # quick fix. get always the 3 first for training and the others 3 for validation set
    # TODO quick fix
    # adaptation_indices[[0,1,2,6,7,8,12,13,14,18,19,20,24,25,26]] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    info = (sequence, ways, shots, iteration, train_task, taskID)

    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # ## plot loss and acc of val task before optimizing
    # if train_task==False:
    #     train_predictions, train_error = forward_pass_for_classifier_training(learner, adaptation_data,
    #                                                                           adaptation_labels, return_scores=True)
    #     train_accuracy = accuracy(train_predictions, adaptation_labels)
    #
    #
    #     val_predictions, val_error = forward_pass_for_classifier_training(learner, evaluation_data,
    #                                                                          evaluation_labels, return_scores=True)
    #     valid_accuracy = accuracy(val_predictions, evaluation_labels)
    #     #plotter.plot(epoch=step, loss=train_error, acc=train_accuracy, split_name='inner', info=info)
    #     #plotter.plot(epoch=step, loss=val_error, acc=val_accuracy, split_name='inner', info=info)
    #     #print('train error {}')

    shuffle = True
    #start_time = time.time()
    if shuffle==True:
        idx = torch.randperm(adaptation_labels.nelement())
        adaptation_labels = adaptation_labels[idx]
        adaptation_data = adaptation_data[idx]

        idx = torch.randperm(evaluation_labels.nelement())
        evaluation_labels = evaluation_labels[idx]
        evaluation_data = evaluation_data[idx]
        #print("--- %s seconds --- [fa] shuffle " % (time.time() - start_time))

    # if train_task==False:
    #     predictions_before, validation_error_before = forward_pass_for_classifier_training(learner, evaluation_data,
    #                                                                          evaluation_labels, ways, return_scores=True)
    #     valid_accuracy_before = accuracy(predictions_before, evaluation_labels)

    # init last layer with the same weights
    if init_last==True:
        #learner.init_last(ways, train=True)
        learner.init_last(ways)
    # Adapt the model
    for step in range(adaptation_steps):
        plot_inner=False
        if (plotter != None) and (iteration%10==0) and task==0 and plot_inner==True:
        #if True:
            train_predictions, train_error = forward_pass_for_classifier_training(learner, adaptation_data, adaptation_labels, ways, return_scores=True, ratio=ratio, train=True)
            train_accuracy = accuracy(train_predictions, adaptation_labels)
            #print(train_accuracy)
            #plotter.plot(epoch=step, loss=train_error, acc=train_accuracy, split_name='inner', info=info)
        else:
            train_error = forward_pass_for_classifier_training(learner, adaptation_data, adaptation_labels, ways, ratio=ratio, train=True)
        #train_error /= len(adaptation_data)
        learner.adapt(train_error)  # Takes a gradient step on the loss and updates the cloned parameters in place
    # get others for validation set and adapt learner
    # get one idx for others, make sure always a different one
    train_error = torch.zeros(1)
    validation_error_others = torch.zeros(1)
    # evaluate on all others
    ID_others = 0
    if True:
        #print('if true')
        #start_time = time.time()
        # others_ = torch.tensor([])
        # others_ = torch.cat((others_, others[0][0]))  # take samples from all MOT sequences
        # for o in others[1][0]:  # go through market and cuhk depending if used for train task or not
        #     if len(o)>0:
        #         others_ = torch.cat((others_, o))

        num_others = len(others)
        start_time = time.time()
        start_time_load = time.time()
        l = DataLoader(others, batch_size=num_others)
        for others in l:
            #print("--- %s seconds --- [fa] just load others" % (time.time() - start_time_load))
            #print('loaded')
            ratio[ID_others] = num_others
            start_time_forward = time.time()
            predictions, validation_error_others = forward_pass_for_classifier_training(learner, others[0], torch.ones(num_others).long().to(device) * ID_others, ways,
                                                                    return_scores=True, ratio=ratio, train=False)
            #print("--- %s seconds --- [fa] just forward pass others" % (time.time() - start_time_forward))
            predictions = predictions.argmax(dim=1)
            correct_others = (predictions == (torch.ones(num_others).to(device).long() * ID_others)).sum().float()
            accuracy_others = correct_others / num_others
            if iteration % 100 == 0 and not train_task:
                print("--- %s seconds --- [fa] load and process others " % (time.time() - start_time))
                print('{}/{} others from other sequences missed'.format(int(num_others - correct_others), num_others))

            #others = torch.cat((others_own, others[0]))
            #print('concat')
        #random.shuffle(others_own)
        #others_own = others_own[:ways*shots*2]
        #num_others = int(others[0].size(0))
        #others = torch.cat((others_own, others.to(device))) if len(others)>1 else others_own
        #others = torch.cat((others_own, others[torch.arange(len(others))])) if len(others)>1 else others_own

        #del others_own
        #num_others = int(others.size(0))
        #evaluation_data_others, evaluation_labels_others = others[0].to(device), others[1].to(device) * ID_others

        #evaluation_data_others, evaluation_labels_others = others.to(device), torch.ones(num_others).long().to(device) * ID_others


        #ID_others = random.randrange(ways)
        #replace = (labels == ID_others)
        #labels = labels + replace * (ways - ID_others)

        #adaptation_indices_others = np.zeros(others[0].size(0), dtype=bool)

        # Separate others into adaptation/evalutation sets
        # if flip_p > 0.0:
        # if False:
        #     num_others = int(others[0].size(0) / 4)  # flip and eval+adapt
        #     adaptation_indices_others[np.arange(0, others[0].size(0), 4)] = True
        #     adaptation_indices_others[np.arange(1, others[0].size(0), 4)] = True
        # else:
        #     num_others = int(others[0].size(0) / 2)
        #     adaptation_indices_others[np.arange(0, others[0].size(0), 2)] = True
        # evaluation_indices_others = torch.from_numpy(~adaptation_indices_others)
        # adaptation_indices_others = torch.from_numpy(adaptation_indices_others)
        #
        # adaptation_data_others, adaptation_labels_others = others[0][adaptation_indices_others], others[1][
        #     adaptation_indices_others] * ID_others
        # evaluation_data_others, evaluation_labels_others = others[0][evaluation_indices_others], others[1][
        #     evaluation_indices_others] * ID_others

        # adaptation_data = torch.cat((adaptation_data, adaptation_data_others))
        # adaptation_labels = torch.cat((adaptation_labels, adaptation_labels_others))
    num_others = len(others_own)
    #start_time = time.time()
    others = torch.cat((evaluation_data, others_own))

    evaluation_labels = torch.cat((evaluation_labels, torch.ones(num_others).long().to(device) * ID_others))
    #print("--- %s seconds --- [fa] cat others own seq " % (time.time() - start_time))
    #logger.debug("--- %s seconds --- for construct others and switch to gpu" % (time.time() - start_time))
    ratio[ID_others] = num_others
    #ratio = torch.cat((ratio, torch.ones(1).to(device)*num_others))

    # if init_last==True:
    #     learner.init_last(ways, train=False)
    # Evaluate the adapted model
    #start_time = time.time()
    predictions, validation_error = forward_pass_for_classifier_training(learner, others, evaluation_labels, ways, return_scores=True, ratio=ratio, train=False)
    #print("--- %s seconds --- [fa] just forward pass task+own " % (time.time() - start_time))
    validation_error = validation_error + validation_error_others if validation_error_others > 0 else validation_error
    # debug val task val set
    # if train_task==False and iteration%10==0:
    #     predictions_debug = predictions.argmax(dim=1).view(evaluation_labels.shape)
    #     print(predictions_debug)
    #     print(evaluation_labels)
    #     print(predictions_debug == evaluation_labels)

    # here accuracy of task and others from own sequence
    # (valid_accuracy_without_zero, valid_accuracy_with_zero, valid_accuracy_just_zero) in case of val task
    valid_accuracy = accuracy(predictions, evaluation_labels, iteration, train_task)
    #print(valid_accuracy)

    if train_task==False:
    #     valid_accuracy = (valid_accuracy, valid_accuracy_before)
        validation_error = (validation_error, validation_error_before)
        valid_accuracy = (valid_accuracy[0], valid_accuracy[1], valid_accuracy[2], accuracy_others)

    return validation_error, valid_accuracy

def statistics(dataset):
    unique_id, counter = np.unique(dataset[0].numpy(), return_counts=True)
    num_id = len(unique_id)
    num_bb = sum(counter)
    samples_per_id, counter_samples = np.unique(counter, return_counts=True)
    print('in total {} unique IDs, and {} BBs in total, av. {} BB/ID  '.format(num_id, num_bb, (num_bb/num_id)))
    #print('in total {} unique IDs, print until 80 samples per ID'.format(num_id))
    # for i in range(len(counter_samples)):
    #     if samples_per_id[i]<80:
    #         print('{} samples per ID: {} times'.format(samples_per_id[i], counter_samples[i]))
    return num_id, num_bb

# 1. version
# def sample_task(tasksets, i_to_dataset, sample, val=False):
#     try:
#         i = random.choice(range(len(tasksets)))  # sample sequence
#         seq = i_to_dataset[i]
#         seq_tasks = tasksets[list(tasksets.keys())[i]]
#         if sample==False and val==False:
#             i=1
#             seq_tasks = tasksets[list(tasksets.keys())[i]]
#             seq = i_to_dataset[i]
#         j = random.choice(range(len(seq_tasks)))  # sample task
#         nways = seq_tasks[j].task_transforms[0].n
#         kshots = int(seq_tasks[j].task_transforms[0].k / 2)
#         batch = seq_tasks[j].sample()
#         return batch, nways, kshots, seq, j
#     except ValueError:
#         nways = seq_tasks[j].task_transforms[0].n
#         kshots = int(seq_tasks[j].task_transforms[0].k / 2)
#         # if len(tasksets)>1:
#         #     print('Problem to sample {} ways and {} shots from {}'.format(nways, kshots, i_to_dataset[i]))
#         # else:
#         #     print('Problem to sample {} ways and {} shots from validation sequence'.format(nways, kshots))
#         return sample_task(tasksets, i_to_dataset, sample, val)
#         # i = random.choice(range(len(tasksets)))
#         # batch = tasksets[i].sample()

# 2. version max recursion depth
# def try_to_sample(taskset):
#     try:
#         batch = taskset.sample()
#         return batch
#     except ValueError:
#         return try_to_sample(taskset)
#
#
# def sample_task(sets, nways, kshots, i_to_dataset, sample, val=False):
#     i = random.choice(range(len(sets)))  # sample sequence
#     seq = i_to_dataset[i]
#     # if sample == False and val == False:
#     #     i = 1
#     #     seq_tasks = tasksets[list(tasksets.keys())[i]]
#     #     seq = i_to_dataset[i]
#     n = random.sample(nways, 1)[0]
#     k = random.sample(kshots, 1)[0]
#     transform = [l2l.data.transforms.FusedNWaysKShots(sets[i], n=n, k=k * 2),
#                  l2l.data.transforms.LoadData(sets[i]),
#                  l2l.data.transforms.RemapLabels(sets[i], shuffle=True)]
#     taskset = l2l.data.TaskDataset(dataset=sets[i],
#                                    task_transforms=transform,
#                                    num_tasks=1000)
#
#         # i = random.choice(range(len(tasksets)))
#         # batch = tasksets[i].sample()
#     batch = try_to_sample(taskset)
#     return batch, n, k, seq, i
def sample_task_val(sets, nways, kshots, i_to_dataset, sample, val=False):
    nways = [3,5,10]
    kshots = [3,20,40]
    i = random.choice(range(len(sets)))  # sample sequence
    seq = i_to_dataset[i]
    # if sample == False and val == False:
    #     i = 1
    #     seq_tasks = tasksets[list(tasksets.keys())[i]]
    #     seq = i_to_dataset[i]
    n = random.sample(nways, 1)[0]
    k = random.sample(kshots, 1)[0]
    transform = [l2l.data.transforms.FusedNWaysKShots(sets[i], n=n, k=k * 2),
                 l2l.data.transforms.LoadData(sets[i]),
                 #l2l.data.transforms.RemapLabels(sets[i], shuffle=True)]
                 l2l.data.transforms.RemapLabels(sets[i], shuffle=False)]
    taskset = l2l.data.TaskDataset(dataset=sets[i],
                                   task_transforms=transform,
                                   num_tasks=1000)

        # i = random.choice(range(len(tasksets)))
        # batch = tasksets[i].sample()
    try:
        batch = taskset.sample()
        return batch, n, k, seq, i
    except ValueError:
        return sample_task(sets, nways, kshots, i_to_dataset, sample, val)

def sample_task(sets, nways, kshots, i_to_dataset, sample, val=False, num_tasks=-1, sample_uniform_DB=False):
    #start_time = time.time()
    if sample_uniform_DB:
        if len(sets)>1:
            j = random.choice(range(3))  # sample which dataset
            if j == 0:
                i = random.choice(range(6))  # which of the MOT sequence
            elif j == 1:
                i = 6
            elif j == 2:
                i = 7
        else:
            i = random.choice(range(len(sets)))  # sample sequence
    else:
        i = random.choice(range(len(sets)))  # sample sequence

    seq = i_to_dataset[i]
    if sample == False and val == False:
        i = 1
        seq = i_to_dataset[i]
    n = random.sample(nways, 1)[0]
    k = random.sample(kshots, 1)[0]
    transform = [l2l.data.transforms.FusedNWaysKShots(sets[i], n=n, k=k * 2),
                 l2l.data.transforms.LoadData(sets[i]),
                 l2l.data.transforms.RemapLabels(sets[i], shuffle=True)]
                 #l2l.data.transforms.RemapLabels(sets[i], shuffle=False)]
    taskset = l2l.data.TaskDataset(dataset=sets[i],
                                   task_transforms=transform,
                                   num_tasks=num_tasks)
                                   #num_tasks=1)

        # i = random.choice(range(len(tasksets)))
        # batch = tasksets[i].sample()
    #print("--- %s seconds --- [in] val prepare task dataset " % (time.time() - start_time))
    try:
        #print('val')
        #start_time = time.time()
        batch = taskset.sample() # val
        #print("--- {} s seconds --- [in] val sample task from seq {} ".format((time.time() - start_time), seq))
        N = n * k * 2
        return batch, n, k, seq, i

        # if others from own sequence included in sampling
        #l = torch.randperm(n).repeat_interleave(k*2)
        #return ((batch[0][:N].to(device), l.to(device)), batch[0][N:,0,:,:,:].to(device)), n, k, seq, i  # 0 hier um nicht die geflippten FM zu bekommen
    except ValueError:
        return sample_task(sets, nways, kshots, i_to_dataset, sample, val, num_tasks)

def sample_task_new(sets, nways, kshots, i_to_dataset, sample, val=False, num_tasks=-1, sample_uniform_DB=False):
    #start_time = time.time()
    if sample_uniform_DB:
        if len(sets)>1:
            j = random.choice(range(7))  # sample which dataset, 0 MOT, 1,2,3 cuhk, 4,5,6 market to balance IDs
            #j = random.choice(range(4))  # sample which dataset, 0 MOT, 1,2,3 cuhk to balance IDs
            if j == 0:
                i = random.choice(range(6))  # which of the MOT sequence
            elif j == 1 or j == 2 or j == 3:
                i = 6
            elif j == 4 or j == 5 or j == 6:
                i = 7
        else:
            i = random.choice(range(len(sets)))  # sample sequence
    else:
        i = random.choice(range(len(sets)))  # sample sequence

    seq = (i_to_dataset[i], i)
    if sample == False and val == False:
        i = 2
        seq = i_to_dataset[i]
    n = random.sample(nways, 1)[0]
    k = random.sample(kshots, 1)[0]
    transform = [l2l.data.transforms.FusedNWaysKShots(sets[i], n=n, k=k * 2),
                 l2l.data.transforms.LoadData(sets[i]),
                 l2l.data.transforms.RemapLabels(sets[i], shuffle=True)]
                 #l2l.data.transforms.RemapLabels(sets[i], shuffle=False)]
    taskset = l2l.data.TaskDataset(dataset=sets[i],
                                   task_transforms=transform,
                                   num_tasks=num_tasks)

        # i = random.choice(range(len(tasksets)))
        # batch = tasksets[i].sample()
    #print("--- %s seconds --- [in] train prepare task dataset " % (time.time() - start_time))
    try:
        #print('train')
        #start_time=time.time()
        batch = taskset.sample() # train
        #print("--- {} s seconds --- [in] train sample task from seq {} ".format((time.time() - start_time), seq))
        #print('taskset samples')
        return batch, n, k, seq, i
        # others in task set gesampelt
        #N = n * k * 2
        #l = torch.randperm(n).repeat_interleave(k * 2)
        #return ((batch[0][:N].to(device), l.to(device)), batch[0][N:,0,:,:,:].to(device)), n, k, seq, i
    except ValueError:
        return sample_task_new(sets, nways, kshots, i_to_dataset, sample, val, num_tasks)

def get_others(sets, i, num_others, use_market, i_to_dataset):
    ## new
    #others_FM = torch.tensor([])
    others_FM = []
    others_FM_val = torch.tensor([])
    #others_ID = torch.tensor([])
    others_ID = []
    others_ID_in_seq = torch.tensor([])

    others_ID_idx = torch.tensor([])
    limit=5000
    for j, s in enumerate(sets):
        ## include train / val split ID wise
        # idx = torch.randperm(s.dataset.data[0].nelement())
        # s = (s.dataset.data[1][idx], s.dataset.data[0][idx])
        # total_num_others = s[1].nelement()
        # output, inv_idx, counts = torch.unique(s[1], return_inverse=True, return_counts=True)
        # num_val_id = int(output.nelement() * 0.3)
        # num_samples = 0
        # #others_ = torch.tensor([]).to(device)
        # for i in range(num_val_id):
        #     num_samples += counts[i]
        #     others_FM_val = torch.cat((others_FM_val, (s[0][(s[1] == output[i])]).float()))
        # #others_val = (others_, torch.ones(others_.shape[0]).long().to(device))
        # #others_ = torch.tensor([]).to(device)
        # for i in range(num_val_id, output.nelement()):
        #     others_FM = torch.cat((others_FM, (s[0][(s[1] == output[i])]).float()))
        # #others = (others_, torch.ones(others_.shape[0]).long().to(device))
        # print('work with {} train samples and {} val samples for set {}'.format(total_num_others-num_samples, num_samples, j))
        # others_ID = torch.cat((others_ID, torch.ones(total_num_others-num_samples) * j))

    # include train / val split easier
    #     if j == 6 or j==7:
    #         total_num = s.dataset.data[0][:-2000].nelement()
    #         val = int(total_num * 0.3)
    #         others_FM_val = torch.cat((others_FM_val, s.dataset.data[1][:val]))
    #         others_FM = torch.cat((others_FM, s.dataset.data[1][val:-2000]))
    #         others_ID = torch.cat((others_ID, torch.ones(s.dataset.data[1][val:-2000].shape[0]) * j))
    #         print('work with {} train samples and {} val samples for set {}'.format(total_num - val,
    #                                                                                 val, j))
    #     else:
    #         total_num = s.dataset.data[0].nelement()
    #         val = int(total_num * 0.3)
    #         others_FM_val = torch.cat((others_FM_val, s.dataset.data[1][:val]))
    #         others_FM = torch.cat((others_FM, s.dataset.data[1][val:]))
    #         others_ID = torch.cat((others_ID, torch.ones(s.dataset.data[1][val:].shape[0])*j))
    #         print('work with {} train samples and {} val samples for set {}'.format(total_num - val,
    #                                                                                 val, j))
    #
    # return (others_FM.to(device), others_ID.to(device), others_FM_val.to(device))


        # without train / val split
        #if j == 6 or j == 7 or s.dataset.data[1].shape[0]>10000:
        if len(s)>limit:
            #i = torch.randperm(int(s.dataset.data[1].shape[0] * 0.5))
            #others_FM = torch.cat((others_FM, s.dataset.data[1][i]))
            #others_ID = torch.cat((others_ID, torch.ones(int(s.dataset.data[1].shape[0] * 0.5)) * j))
            # random.shuffle(s.dataset.data[1])
            # others_FM = torch.cat((others_FM, s.dataset.data[1][:5000]))
            random.shuffle(s.dataset.fm)
            others_FM.append(s.dataset.fm[:limit])
            others_ID.append(torch.ones(limit)*j)
            print('work with {} samples and for set {}'.format(5000, i_to_dataset[j]))
        else:
            # others_FM = torch.cat((others_FM, s.dataset.fm))
            # others_ID_idx = torch.cat((others_ID_idx, torch.ones(s.dataset.data[1].shape[0])*j))
            # others_ID = torch.cat((others_ID, s.dataset.data[0].float()))
            others_FM.append(s.dataset.fm)
            others_ID.append(torch.ones(len(s.dataset))*j)
            print('work with {} samples and for set {}'.format(len(s), i_to_dataset[j]))
        #others_ID_in_seq = torch.cat((others_ID_in_seq, s.dataset.data[0].float()))
    #return (others_FM.to(device), others_ID, others_ID_in_seq)

    others_FM = torch.cat(others_FM)
    others_ID = torch.cat(others_ID)
    print('in total {} others'.format(others_FM.shape[0]))
    #return (others_FM.to(device), others_ID
    return others_FM, others_ID
    #return (others_FM, others_ID)

    ## OLD too long
    #use_market = True
    use_cuhk = True
    if use_market:
        set = sets[-1]
        others_FM = set.dataset.data[1]
        #others_FM = torch.tensor(set.dataset.data[1]).to(device)

        #others_ID = set.dataset.data[0] ## to seperate IDs
        others_ID = torch.ones(others_FM.shape[0]).long()*7
        #others_ID = torch.tensor([])

        if use_cuhk:
            set = sets[-2]
            others_FM =torch.cat((set.dataset.data[1], others_FM))
            others_ID = torch.cat((torch.ones(set.dataset.data[1].shape[0]).long()*6, others_ID))
            #others_ID = torch.tensor([])
        return (others_FM, others_ID)

    N = num_others
    others_FM = torch.tensor([])
    sets = [s for j, s in enumerate(sets) if j != i]  # exclude the set used for N ways
    if N > 0: # take specific number of samples
        for _ in range(N):
            j = random.randint(0,len(sets)-1)
            k = random.randint(0,len(sets[j])-1)
            others_FM = torch.cat((others_FM, sets[j].dataset[k][0]))
            try:
                others_FM = torch.cat((others_FM, sets[j].dataset[k+1][0]))
            except:
                others_FM = torch.cat((others_FM, sets[j].dataset[k-1][0]))
    else:
        # -1 take all possible samples from MOT sequences
        if i >= 6:  # tasks come from either market or cuhk
            end = -1
        else:
            end = -2
        sets = sets[:end]
        for s in sets:
            others_FM = torch.cat((others_FM, s.dataset.data[1]))

    #others_ID = torch.ones(others_FM.shape[0]).long()
    others_ID = torch.tensor([])
    return (others_FM, others_ID)


@ex.automain
def my_main(_config, reid, _run):
    sacred.commands.print_config(_run)

    sampled_val = {}
    sampled_train = {}

    # set all seeds
    torch.manual_seed(reid['seed'])
    torch.cuda.manual_seed(reid['seed'])
    np.random.seed(reid['seed'])
    torch.backends.cudnn.deterministic = True
    random.seed = (reid['seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #print(_config)

    output_dir = osp.join(get_output_dir(reid['module_name']), reid['name'])
    tb_dir = osp.join(get_tb_dir(reid['module_name']), reid['name'])

    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    #########################
    # Initialize dataloader #
    #########################
    print("[*] Initializing Dataloader")
    start_time = time.time()

    if reid['dataloader']['build_dataset']:
        db_train = Datasets(reid['db_train'], reid['dataloader'])
        print('Build trainings set, finish ...')
        sys.exit()
    else:
        start_time_load = time.time()
        i_to_dataset = {}
        with h5py.File('./data/ML_dataset/{}.h5'.format(reid['ML']['db']), 'r') as hf:
            datasets = list(hf.keys())
            datasets = [d for d in datasets if d != reid['dataloader']['validation_sequence']]
            #datasets = [d for d in datasets if d not in ['MOT-02', 'MOT-04', 'MOT-05', 'MOT-09', 'MOT-10', 'MOT-11', 'MOT-13']]
            #datasets = [d for d in datasets if d == 'cuhk03' or d == 'market1501']
            for  i,d in enumerate(datasets):
                i_to_dataset[i] = d
            print('Train with {} and use {} as validation set'.format(datasets, reid['dataloader']['validation_sequence']))

            data = {}
            bb_total = 0
            ids_total = 0
            for set in datasets:
                seq = hf.get(set)
                d, l = seq.items()
                data[set] = ((torch.tensor(l[1]), torch.tensor(d[1])))
                print('loaded {}'.format(set))
                ids, bb = statistics(data[set])
                ids_total += ids
                bb_total += bb
                # for i in data[set][0]:
                #     print(i)
            print(
                'OVERALL: {} unique IDs, and {} BBs in total, av. {} BB/ID  '.format(ids_total, bb_total, (bb_total / ids_total)))

            if reid['dataloader']['validation_sequence'] != 'None':
                validation_data = hf.get(reid['dataloader']['validation_sequence'])
                d, l = validation_data.items()
                validation_data = ((torch.tensor(l[1]), torch.tensor(d[1])))
                print('loaded validation {}'.format(reid['dataloader']['validation_sequence']))
                #statistics(validation_data)
            print("--- %s seconds --- for loading hdf5 database" % (time.time() - start_time_load))

    start_time_tasks = time.time()
    #nways = reid['ML']['nways']
    #nways_list = [3,5,10]
    if reid['ML']['range']:
        nways_list = list(range(2, reid['ML']['nways']+1))
        kshots_list = list(range(1, reid['ML']['kshots']+1))
    else:
        nways_list = reid['ML']['nways']
        if type(nways_list) is int:
            nways_list = [nways_list]
        kshots_list = reid['ML']['kshots']
        if type(kshots_list) is int:
            kshots_list = [kshots_list]
    num_tasks = reid['ML']['num_tasks']
    num_tasks_val = reid['ML']['num_tasks_val']
    adaptation_steps = reid['ML']['adaptation_steps']
    meta_batch_size = reid['ML']['meta_batch_size']

    if reid['dataloader']['validation_sequence'] != 'None':
        validation_set = ML_dataset(validation_data, reid['ML']['flip_p'])
        validation_set = [l2l.data.MetaDataset(validation_set)]
    else:
        validation_set = []

    # val_transform = []
    # for i in nways_list:
    #     for kshots in kshots_list:
    #         val_transform.append([l2l.data.transforms.FusedNWaysKShots(validation_set, n=i, k=kshots * 2),
    #                               l2l.data.transforms.LoadData(validation_set),
    #                               l2l.data.transforms.RemapLabels(validation_set, shuffle=True)
    #                               ])
    #
    # val_taskset = {}
    # for t in val_transform:
    #     if 'val' not in val_taskset.keys():
    #         val_taskset['val']=[l2l.data.TaskDataset(dataset=validation_set,
    #                                                 task_transforms=t,
    #                                                 num_tasks=num_tasks_val)]
    #     else:
    #         val_taskset['val'].append(l2l.data.TaskDataset(dataset=validation_set,
    #                                              task_transforms=t,
    #                                              num_tasks=num_tasks_val))

    # val_taskset = l2l.data.TaskDataset(dataset=validation_set,
    #                                      task_transforms=val_transform,
    #                                      num_tasks=num_tasks_val)

    sequences = []
    for dataset in data:
        sequences.append(ML_dataset(data[dataset], reid['ML']['flip_p']))

    meta_datasets = []
    #transforms = []
    #tasksets = []
    #tasksets = {}

    for i, s in enumerate(sequences):
        meta_datasets.append(l2l.data.MetaDataset(s))
        # for j in nways_list:
        #     for kshots in kshots_list:
        #         transform = [l2l.data.transforms.FusedNWaysKShots(meta_datasets, n=j, k=kshots*2),
        #             l2l.data.transforms.LoadData(meta_datasets),
        #              l2l.data.transforms.RemapLabels(meta_datasets, shuffle=True)]
        #         if i not in tasksets.keys():
        #             tasksets[i] = [l2l.data.TaskDataset(dataset=meta_datasets,
        #                                              task_transforms=transform,
        #                                              num_tasks=num_tasks)]
        #         else:
        #             tasksets[i].append(l2l.data.TaskDataset(dataset=meta_datasets,
        #                                              task_transforms=transform,
        #                                              num_tasks=num_tasks))

    print("--- %s seconds --- for construction of meta-datasets" % (time.time() - start_time_tasks))
    print("--- %s seconds --- for loading db and building meta-datasets " % (time.time() - start_time))
    
    ##########################
    # Clean #
    ##########################
    del data
    del validation_data
    #del meta_datasets
    #del validation_set
    del sequences
    del d
    del seq
    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building CNN")
    print('remap on my own because of others own')

    obj_detect = FRCNN_FPN(num_classes=2).to(device)
    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                                          map_location=lambda storage, loc: storage))

    #box_head_classification = obj_detect.roi_heads.box_head
    #box_predictor_classification = obj_detect.roi_heads.box_predictor
    #reID_network = reID_Model(box_head_classification, box_predictor_classification, n_list=nways_list)

    reID_network = reID_Model(obj_detect.roi_heads.box_head, obj_detect.roi_heads.box_predictor, n_list=nways_list)

    reID_network.train()
    reID_network.cuda()
    #reID_network.last_layers_freeze()

    # for n in nways_list:
    #     if n>2:
    #         for p in reID_network.additional_layer[n].parameters():
    #             p.requires_grad = False
    lr = float(reid['solver']['LR_init'])
    if reid['ML']['maml']:
        model = MAML(reID_network, lr=1e-3, first_order=True, allow_nograd=True)
        #opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        opt = torch.optim.Adam([
            {'params':model.module.head.parameters()},
            {'params':model.module.predictor.parameters()}
        ], lr=lr)
    if reid['ML']['learn_LR']:
        model = MetaSGD(reID_network, lr=1e-3, first_order=True, allow_nograd=True, global_LR=reid['ML']['global_LR'])
        #opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_lr = float(reid['solver']['LR_LR'])
        opt = torch.optim.Adam([
            {'params':model.module.head.parameters()},
            {'params':model.module.predictor.parameters()},
            {'params':model.lrs, 'lr': lr_lr}
        ], lr=lr)
        #model.last_layers_freeze()
    # for p in model.parameters():
    #     a = p.numel


    if reid['solver']['continue_training']:
        if os.path.isfile(reid['solver']['checkpoint']):
            print("=> loading checkpoint '{}'".format(reid['solver']['checkpoint']))
            checkpoint = torch.load(reid['solver']['checkpoint'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(reid['solver']['checkpoint'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # for p in model.parameters():
    #     a = p.numel
    # opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    # print(sum(p.numel() for p in box_head.parameters() if p.requires_grad))



    ##################
    # Begin training #
    del reID_network
    del obj_detect
    ##################
    print("[*] Training ...")
    clamping_LR = True
    if clamping_LR:
        print('clamp LR < 0')
    #print('!!!! sample val differently')
    losses_meta_train = AverageMeter('Loss', ':.4e')
    losses_meta_val = AverageMeter('Loss', ':.4e')
    acc_meta_train = AverageMeter('Acc', ':6.2f')
    acc_meta_val = AverageMeter('Acc', ':6.2f')
    acc_zero_meta_val = AverageMeter('Acc', ':6.2f')
    acc_just_zero_meta_val = AverageMeter('Acc', ':6.2f')
    acc_others_meta_val = AverageMeter('Acc', ':6.2f')

    # how to sample, uniform from the 3 db or uniform over sequences
    sample_db = reid['ML']['db_sample_uniform']

    # get others for market and cuhk
    #start_time = time.time()
    #others, others_id, others_val  = get_others(meta_datasets, -1, reid['ML']['num_others'], use_market=True)

    #others, others_id  = get_others(meta_datasets, -1, reid['ML']['num_others'], True, i_to_dataset)
    #others = l2l.data.UnionMetaDataset(meta_datasets)
    #start_time = time.time()
    #others_val = get_others(meta_datasets, -1, reid['ML']['num_others'], use_market=False)
    #logger.debug("--- %s seconds --- for get others" % (time.time() - start_time))
    #others_val = (others_val, others)

    if reid['ML']['market_others']:
        others = get_others(meta_datasets, -1, reid['ML']['num_others'], use_market=True)
        ## split others in train and val, shuffle first
        idx = torch.randperm(others[1].nelement())
        others = (others[0][idx], others[1][idx])
        total_num_others = others[1].nelement()
        output, inv_idx, counts = torch.unique(others[1], return_inverse=True, return_counts=True)
        num_val_id = int(output.nelement() * 0.3)
        num_samples = 0
        others_ = torch.tensor([]).to(device)
        for i in range(num_val_id):
            num_samples += counts[i]
            others_= torch.cat((others_, (others[0][(others[1] == output[i])]).float()))
        others_val = (others_, torch.ones(others_.shape[0]).long().to(device))
        others_ = torch.tensor([]).to(device)
        for i in range(num_val_id, output.nelement()):
            others_ = torch.cat((others_, (others[0][(others[1] == output[i])]).float()))
        others = (others_, torch.ones(others_.shape[0]).long().to(device))
        print('work with {} train samples and {} val samples for others'.format(total_num_others-num_samples, num_samples))

    plotter=None
    if reid['solver']["plot_training_curves"]:
        now = datetime.datetime.now()
        run_name = now.strftime("%Y-%m-%d_%H:%M")
        plotter = VisdomLinePlotter_ML(env=run_name, offline=reid['solver']['plot_offline'],
                                       info=(nways_list, kshots_list, meta_batch_size, num_tasks,
                                             num_tasks_val, reid['dataloader']['validation_sequence'],
                                             reid['ML']['flip_p']))

    # safe model 10 times
    safe_every = int(_config['reid']['solver']['iterations'] / 10)
    init_last = reid['ML']['init_last']
    for iteration in range(1,_config['reid']['solver']['iterations']+1):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        # without zero
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        # with zero
        meta_valid_error_before = 0.0
        meta_valid_accuracy_before = 0.0
        # just zero
        meta_valid_accuracy_just_zero = 0.0
        meta_valid_accuracy_others = 0.0

        # for lr in model.lrs:
        #     lr_mask = (lr > 0)
        #     anzahl_false = (lr < 0).sum().item()
        #     print('teste model oben')
        #     if anzahl_false > 0:
        #         print('problem {} kleiner 0'.format(anzahl_false))
        #         print((anzahl_false))

        for task in range(meta_batch_size):
            # Compute meta-validation loss
            # here no backward in outer loop, just inner
            if len(validation_set)>0:
                learner = model.clone()

                # for lr in learner.lrs:
                #     lr_mask = (lr > 0)
                #     anzahl_false = (lr < 0).sum().item()
                #     print('teste learner val')
                #     if anzahl_false > 0:
                #         print('problem {} kleiner 0'.format(anzahl_false))
                #         print((anzahl_false))


                #batch, nways, kshots, _, taskID = sample_task_val(validation_set, nways_list, kshots_list, i_to_dataset, reid['ML']['sample_from_all'], val=True)
                #start_time = time.time()
                batch, nways, kshots, _, taskID = sample_task(validation_set, nways_list, kshots_list, i_to_dataset,
                                                              reid['ML']['sample_from_all'], val=True, num_tasks=num_tasks_val)
                #N=nways*kshots*2  # because of adaptation+evaluation data
                #not_used_IDs = batch[0][N:]  # from current train task sequence
                #batch = (batch[0][:N], batch[1][:N])
                #batch, used_labels = batch
                # others_val = torch.tensor([]).to(device)
                # others_val_labels = [l for l in validation_set[0].labels if l not in not_used_IDs]
                # for l in others_val_labels:
                #     idx = validation_set[0].labels_to_indices[l]
                #     others_val = torch.cat((others_val, validation_set[0][idx][0][0,:,:,:]))

                batch, used_labels = batch
                others_val_idx = [
                    idx for l, idx in validation_set[0].labels_to_indices.items()
                    if l not in used_labels]
                others_val_idx = functools.reduce(operator.iconcat, others_val_idx, [])
                others_val = validation_set[0][others_val_idx][0]
                # others_val = [validation_set[0][idx][0][0, :, :, :]
                #               for l, idx in validation_set[0].labels_to_indices.items()
                #               if l not in used_labels]
                # others_val = torch.cat(others_val)
                #print("--- %s seconds --- [out] sample val task " % (time.time() - start_time))
                #print('sampled val task')
                # if not reid['ML']['market_others']:
                #     start_time = time.time()
                #     others_val = get_others(meta_datasets, -1, reid['ML']['num_others'], use_market=False)
                #     print("--- %s seconds --- for get others VAL" % ( time.time() - start_time))
                if (nways, kshots) not in sampled_val.keys():
                    sampled_val[(nways, kshots)] = 1
                else:
                    sampled_val[(nways, kshots)] += 1

                others = l2l.data.UnionMetaDataset(meta_datasets)
                #print('before fast adapt val')
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   adaptation_steps,
                                                                   kshots,
                                                                   nways,
                                                                   device,
                                                                   False,
                                                                   plotter,
                                                                   iteration,
                                                                   reid['dataloader']['validation_sequence'],
                                                                   task,
                                                                   taskID,
                                                                   reid,
                                                                   init_last,
                                                                   #others_val)
                                                                   #torch.cat((others, not_used_IDs[:,0,:,:,:].to(device)))
                                                                   #not_used_IDs[:,0,:,:,:].to(device)
                                                                   others_val,
                                                                   others
                                                                   )

                evaluation_error, evaluation_error_before = evaluation_error
                evaluation_accuracy, evaluation_accuracy_before, evaluation_accuracy_just_zero, evaluation_accuracy_others = evaluation_accuracy

                meta_valid_error_before += evaluation_error_before.item()
                meta_valid_accuracy_before += evaluation_accuracy_before.item()  # accuracy with zeros class (meaning others from own sequence)

                meta_valid_accuracy_just_zero += evaluation_accuracy_just_zero.item()  # track acc just zero class
                meta_valid_accuracy_others += evaluation_accuracy_others.item()  # track acc others from other sequence

                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()  # accuracy without zero class

                losses_meta_val.update(evaluation_error.item())
                acc_meta_val.update(evaluation_accuracy.item())
                acc_zero_meta_val.update(evaluation_accuracy_before.item())
                acc_just_zero_meta_val.update(evaluation_accuracy_just_zero.item()) # others from own sequence
                acc_others_meta_val.update(evaluation_accuracy_others.item()) # others from others sequences


            # Compute meta-training loss
            learner = model.clone()  #back-propagating losses on the cloned module will populate the buffers of the original module
            #start_time = time.time()
            batch, nways, kshots, sequence, taskID = sample_task_new(meta_datasets, nways_list, kshots_list, i_to_dataset,
                                                                 reid['ML']['sample_from_all'], num_tasks=num_tasks,
                                                                 sample_uniform_DB=sample_db)
            #print('train samples back in main')
            sequence, sequence_idx = sequence
            #batch, not_used_IDs = batch

            #N = nways * kshots * 2
            #not_used_IDs = batch[0][N:]
            #batch = (batch[0][:N], batch[1][:N])
            # for i in used_IDs_train:
            #     others_seq = others_id_in_seq[others_id == sequence_idx]
               # use =
            batch, used_labels = batch
            # others_val = [meta_datasets[sequence_idx][idx][0][0, :, :, :]
            #               for l, idx in meta_datasets[sequence_idx].labels_to_indices.items()
            #               if l not in used_labels]
            # others_val = torch.cat(others_val)
            others_val_idx = [
                idx for l, idx in meta_datasets[sequence_idx].labels_to_indices.items()
                if l not in used_labels]
            others_val_idx = functools.reduce(operator.iconcat, others_val_idx, [])
            others_val = meta_datasets[sequence_idx][others_val_idx][0]

            # others_val = torch.tensor([]).to(device)
            # others_val_labels = [l for l in meta_datasets[sequence_idx].labels if l not in not_used_IDs]
            # for l in others_val_labels:
            #     idx = meta_datasets[sequence_idx].labels_to_indices[l]
            #     others_val = torch.cat((others_val, meta_datasets[sequence_idx][idx][0][0, :, :, :]))
            #print("--- %s seconds --- [out] sample train task " % (time.time() - start_time))
            #others_seq =
            #print('samples train taks')
            if sequence not in sampled_train.keys():
                sampled_train[sequence] = {}

            if (nways, kshots) not in sampled_train[sequence].keys():
                sampled_train[sequence][(nways, kshots)] = 1
            else:
                sampled_train[sequence][(nways, kshots)] += 1
            # if not reid['ML']['market_others']:
            #     start_time = time.time()
            #     others_train = get_others(meta_datasets, sequence_idx, reid['ML']['num_others'], use_market=False)
            #     print("--- %s seconds --- for get others train" % (time.time() - start_time))
            # if sequence_idx == 6:
            #     others_ = (others_train, ((others[0][1],torch.tensor([])), torch.tensor([])))
            # elif sequence_idx == 7:
            #     others_ = (others_train, ((others[0][0],torch.tensor([])), torch.tensor([])))
            # else:
            #     others_ = (others_train, others)
            #start_time = time.time()
            #others_train = others[others_id!=sequence_idx]
            #logger.debug("--- %s seconds --- for get others train" % (time.time() - start_time))
            #print('before fast adapt train')
            #print('before fast adapt')
            others = l2l.data.UnionMetaDataset(
                [m for i,m in enumerate(meta_datasets) if i!=sequence_idx]
            )
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               adaptation_steps,
                                                               kshots,
                                                               nways,
                                                               device,
                                                               True,
                                                               plotter,
                                                               iteration,
                                                               sequence,
                                                               task,
                                                               taskID,
                                                               reid,
                                                               init_last,
                                                               #torch.cat((others[others_id!=sequence_idx], not_used_IDs[:,0,:,:,:].to(device)))
                                                               #not_used_IDs[:, 0, :, :, :].to(device)
                                                               others_val,
                                                               #others[others_id != sequence_idx]
                                                               others
                                                               )
            #print('after fast adapt train')
            evaluation_error.backward()  # compute gradients, populate grad buffers of maml
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            losses_meta_train.update(evaluation_error.item())
            acc_meta_train.update(evaluation_accuracy.item())

            evaluation_error = torch.zeros(1)

        # update the best value for loss
        if len(validation_set)>0:
            safe_best_loss = losses_meta_val.update_best(losses_meta_val.avg, iteration)
        else:
            safe_best_loss = losses_meta_val.update_best(losses_meta_train.avg, iteration)

        if iteration%10==0:
            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Loss', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Loss', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy - on task', meta_valid_accuracy / meta_batch_size)
            print('Meta Valid Accuracy - on task with others from own sequence', meta_valid_accuracy_before / meta_batch_size)
            print('Meta Valid Accuracy - just others from own sequence', meta_valid_accuracy_just_zero / meta_batch_size)
            print('Meta Valid Accuracy - just others from other sequences', meta_valid_accuracy_just_zero / meta_batch_size)
            print('\n')
            print('Mean Meta Train Loss', losses_meta_train.avg)
            print('Mean Meta Train Accuracy', acc_meta_train.avg)
            print('Mean Meta Val Loss', losses_meta_val.avg)
            print('Mean Meta Val Accuracy - on task', acc_meta_val.avg)
            print('Mean Meta Val Accuracy - on task with others from own sequence', acc_zero_meta_val.avg)
            print('Mean Meta Val Accuracy - just others from own sequence', acc_just_zero_meta_val.avg)
            print('Mean Meta Val Accuracy - just others from other sequences', acc_others_meta_val.avg)

            if reid['ML']['learn_LR'] and reid['ML']['global_LR']:
                for p in model.lrs:
                    print('LR: {}'.format(p.item()))
                    #print('LR: {}'.format(p))
            # print('safe state dict')
            # model = osp.join(output_dir, 'reID_Network.pth')
            # torch.save(maml.state_dict(), model)
            # print('sampled tasks from train {}'.format(sampled_train))
            # print('sampled tasks from val {}'.format(sampled_val))

        #if iteration>30000 and (iteration%safe_every==0 or safe_best_loss==1) and (iteration%100==0):
        if iteration>500 and (iteration%safe_every==0 or safe_best_loss==1):
            if iteration%10==0:
                model_s = '{}_reID_Network.pth'.format(iteration)
                if safe_best_loss:
                    model_s = 'best_reID_Network.pth'
                    # if reid['ML']['learn_LR']:
                    #     for p in model.lrs:
                    #         print('LR: {}'.format(p.item()))

                #print('safe state dict {}'.format(model_s))
                model_name = osp.join(output_dir, model_s)
                #torch.save(model.state_dict(), model_name)

                state_dict_to_safe = remove_last(model.state_dict(), model.module.num_output)
                save_checkpoint({
                    'epoch': iteration,
                    'state_dict': state_dict_to_safe,
                    'best_acc': (acc_meta_train.avg, acc_meta_val.avg),
                    #'optimizer': opt.state_dict(),
                    'optimizer': None,
                }, filename=model_name)

        if reid['solver']["plot_training_curves"] and (iteration%100==0 or iteration==1):
            if reid['ML']['learn_LR'] and reid['ML']['global_LR'] :
                plotter.plot(epoch=iteration, loss=meta_train_error/meta_batch_size,
                             acc=meta_train_accuracy/meta_batch_size, split_name='train_task_val_set', LR=model.lrs[0])
            else:
                plotter.plot(epoch=iteration, loss=meta_train_error / meta_batch_size,
                             acc=meta_train_accuracy / meta_batch_size, split_name='train_task_val_set')
            plotter.plot(epoch=iteration, loss=losses_meta_train.avg, acc=acc_meta_train.avg, split_name='train_task_val_set MEAN')
            plotter.plot(epoch=iteration, loss=meta_valid_error/meta_batch_size, acc=meta_valid_accuracy/meta_batch_size, split_name='val_task_val_set')
            plotter.plot(epoch=iteration, loss=losses_meta_val.avg, acc=acc_meta_val.avg, split_name='val_task_val_set MEAN')
            #plotter.plot(epoch=iteration, loss=meta_valid_error_before/meta_batch_size, acc=meta_valid_accuracy_before/meta_batch_size, split_name='val_task_val_set_before')
        # Average the accumulated gradients and optimize
        #diff_params = [p for p in model.parameters() if p.requires_grad]
        for p in [p for p in model.parameters() if p.requires_grad]:
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_batch_size)

        opt.step()

        if clamping_LR:
            #print('hier clamping')
            for i, lr in enumerate(model.lrs):
                if (lr < 0).sum().item() > 0 :
                    lr_mask = (lr > 0)
                    #print('in lrs {} sind {} < 0'.format(i, anzahl_false))
                    model.lrs[i] = torch.nn.Parameter(lr*lr_mask)


    print('sampled tasks from train {}'.format(sampled_train))
    print('sampled tasks from val {}'.format(sampled_val))

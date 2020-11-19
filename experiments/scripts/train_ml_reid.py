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
from torchviz import make_dot

#from tracktor.transforms import FusedNWaysKShots

ex = Experiment()
ex.add_config('experiments/cfgs/ML_reid.yaml')
# if torch.cuda.device_count() > 1:
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# else:
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
            model_parameters = [p for name, p in model.named_parameters() if p.requires_grad and 'lrs' not in name]
            #model_parameters = [p for name, p in model.named_parameters() if p.requires_grad]
            #for p, lr, g in zip(model.parameters(), lrs, grads):
            for p, lr, g in zip(model_parameters, lrs, grads):
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
    def __init__(self, model, lr=1.0, first_order=False, allow_unused=None, allow_nograd=False, lrs=None, lrs_inner=None,
                 global_LR=False, others_neuron_b=None, others_neuron_w=None, template_neuron_b=None,
                 template_neuron_w=None):
        super(l2l.algorithms.MetaSGD, self).__init__()
        self.module = model
        self.module.global_LR = global_LR
        # LR per layer
        # if lrs is None:
        #     lrs = [torch.ones(1).to(device) * lr for p in model.parameters()]
        #     lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])

        self.others_neuron_weight = others_neuron_w
        self.others_neuron_bias = others_neuron_b
        self.template_neuron_weight = template_neuron_w
        self.template_neuron_bias = template_neuron_b

        if global_LR:
            # one global LR
            if lrs is None:
                lrs = [torch.nn.Parameter(torch.ones(1).to(device) * lr)]
            self.lrs = lrs[0]
        else:
            if lrs is None:
                # add LRs for template (outer model)
                lrs = [torch.ones_like(p) * lr for name, p in self.named_parameters()
                       if 'last' not in name and 'head' not in name]
                #print('quick fix to exclude LRs for additional last_N layer')
                #lrs = lrs[:-2]  # quick fix to exclude LRs for additional last_N layer
                lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
                self.lrs = lrs

                # add LRs for inner model
                lrs = [torch.ones_like(p) * lr for name, p in self.module.named_parameters()]
                lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
                self.module.lrs = lrs

            else:
                lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
                self.lrs = lrs
                lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs_inner])
                self.module.lrs = lrs

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
                       lrs_inner=clone_parameters(self.module.lrs),
                       first_order=first_order,
                       allow_unused=allow_unused,
                       allow_nograd=allow_nograd,
                       global_LR=self.module.global_LR,
                       others_neuron_b=self.others_neuron_bias.clone(),
                       others_neuron_w=self.others_neuron_weight.clone(),
                       template_neuron_b=self.template_neuron_bias.clone(),
                       template_neuron_w=self.others_neuron_weight.clone())

    def adapt(self, loss, first_order=None, allow_nograd=False):
        """
        **Descritpion**
        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
        if first_order is None:
            first_order = self.first_order

        second_order = not first_order
        #diff_params = [p for p in self.module.parameters() if p.requires_grad]
        diff_params = [p for name, p in self.module.named_parameters() if p.requires_grad and 'lrs' not in name]
        #diff_params = [p for name, p in self.module.named_parameters() if p.requires_grad]
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

        self.module = meta_sgd_update(self.module, self.module.lrs, gradients)

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
            # LR per Parameter
            last = ways + 1  # always others neuron + ways times duplicated template neuron
            last_n = f"last_{last}"

            # repeat template neuron and LRs
            repeated_bias = self.template_neuron_bias.repeat(ways)
            repeated_weights = self.template_neuron_weight.repeat(ways, 1)
            repeated_weights_lr = self.lrs[2].repeat(ways, 1)
            repeated_bias_lr = self.lrs[3].repeat(ways)

            repeated_weights = torch.cat((self.others_neuron_weight, repeated_weights))
            repeated_bias = torch.cat((self.others_neuron_bias, repeated_bias))
            repeated_weights_lr = torch.cat((self.lrs[0], repeated_weights_lr))
            repeated_bias_lr = torch.cat((self.lrs[1], repeated_bias_lr))
            for name, layer in self.module.named_modules():
                if name == last_n:
                    for param_key in layer._parameters:
                        if param_key == 'weight':
                            layer._parameters[param_key] = repeated_weights
                            #self.lrs.append(repeated_weights_lr)
                            self.module.lrs._parameters['4'] = repeated_weights_lr
                        elif param_key == 'bias':
                            layer._parameters[param_key] = repeated_bias
                            #self.lrs.append(repeated_bias_lr)
                            self.module.lrs._parameters['5'] = repeated_bias_lr
                    break

class ML_dataset(Dataset):
    def __init__(self, fm, id, flip_p=-1):
        self.fm = fm
        self.id = id
        self.flip_p = flip_p

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
            features = self.fm[1][item].flip(-1).unsqueeze(0)
            features = torch.cat((self.fm[1][item].unsqueeze(0), features))

            label = self.id[0][item] #.unsqueeze(0)
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
        for n in n_list:
            n += 1
            self.add_module(f"last_{n}", torch.nn.Linear(1024, n).to(device))

        self.num_output = n_list

    def forward(self, x, nways, train):
        feat = self.head(x)
        last_n = f"last_{nways + 1}"

        # because of variable last name
        for name, layer in self.named_modules():
            if name == last_n:
                x = layer(feat)
        return x


def remove_last(state_dict, num_output):
    r = dict(state_dict)
    for n in num_output:
        key_weight = f"module.last_{n+1}.weight"
        key_bias = f"module.last_{n+1}.bias"
        del r[key_weight]
        del r[key_bias]
    return r


def forward_pass_for_classifier_training(learner, features, scores, nways, eval=False, return_scores=False, ratio=[], train=False, t=-1):
    #ratio = torch.sqrt(ratio)
    if eval:
        learner.eval()
    if type(features) is tuple:
        #start_time_over = time.time()
        #start_time = time.time()
        # first task IDs, than others own seq
        tasks_others_own = torch.cat((features[0], features[1]))
        #print('time to concat task and others own {}'.format(time.time()-start_time))

        if len(features[2]) > 1:
            # third others from all others
            # start_time = time.time()
            class_logits2 = learner(tasks_others_own, nways, train)
            # print('first forward pass {}'.format(time.time()-start_time))
            # start_time = time.time()
            class_logits = learner(features[2], nways, train)
            #print('second forward pass {}'.format(time.time() - start_time))
            #start_time=time.time()
            class_logits = torch.cat((class_logits2, class_logits))
            #print('concat logits {}'.format(time.time() - start_time))
            #print('OVERALL forward {}'.format(time.time() - start_time_over))
        else:
            class_logits = learner(tasks_others_own, nways, train)
    else:
        start_time = time.time()
        class_logits = learner(features, nways, train)
        if not train:
            t2=time.time() - start_time
            print('forward pass to get logits with concat {}'.format(t2))
            print('OVERALL forward {}'.format(t+t2))

    if return_scores:
        pred_scores = F.softmax(class_logits, -1)
        if not train:
            w = torch.ones(int(torch.max(scores).item()) + 1).to(device)
            # w[0] = w[0]*ratio  # downweight 0 class
            # w[1:] = w[1:]*(1/ratio)  # upweight inactive
            w = w * (1 / ratio)
            #loss = F.cross_entropy(class_logits, scores.long(), weight=w, reduction='sum')
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


def accuracy(predictions, targets, iter=-1, train=False, seq=-1, num_others_own=-1):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    #if not train:
        #print(predictions)
    compare = (predictions == targets)

    non_zero_target = (targets != 0)
    num_non_zero_targets = non_zero_target.sum()

    zero_target = (targets == 0)
    num_zero_targets = zero_target.sum()

    correct_others_own = compare[num_non_zero_targets:(num_non_zero_targets+num_others_own)].sum()
    valid_accuracy_others_own = correct_others_own.float() / num_others_own
    if iter % 10 == 0:
        name = 'train '+seq if train else 'val'

        #num_correct_predictions_zero_targets = compare[num_non_zero_targets:].sum()
        num_correct_predictions_zero_targets = compare[targets == 0].sum()
        #assert num_correct_predictions_zero_targets == num_correct_predictions_zero_targets2
        zero_target_missed = num_zero_targets - num_correct_predictions_zero_targets

        #num_correct_predictions_non_zero_targets = compare[:num_non_zero_targets].sum()
        num_correct_predictions_non_zero_targets = compare[targets != 0].sum()
        #assert num_correct_predictions_non_zero_targets == num_correct_predictions_non_zero_targets2
        non_zero_target_missed = num_non_zero_targets - num_correct_predictions_non_zero_targets
        print(f"{name:<20} {non_zero_target_missed}/{num_non_zero_targets} persons missed, "
              f"{zero_target_missed}/{num_zero_targets} others missed, "
              f"{num_others_own-correct_others_own}/{num_others_own} others OWN sequence missed")

    #valid_accuracy_with_zero = (predictions == targets).sum().float() / targets.size(0)
    valid_accuracy_with_zero = compare.sum().float() / targets.size(0)
    # if torch.abs(valid_accuracy_with_zero - valid_accuracy_with_zero2) > 1e-4 :
    #     print('ALL problem ist {} nicht gleich {}'.format(valid_accuracy_with_zero, valid_accuracy_with_zero2))
    #     exit()

    #valid_accuracy_without_zero = (predictions[:num_non_zero_targets] == targets[:num_non_zero_targets]).sum().float() / targets[:num_non_zero_targets].size(0)
    valid_accuracy_without_zero = compare[non_zero_target].sum().float() / num_non_zero_targets
    # if torch.abs(valid_accuracy_without_zero - valid_accuracy_without_zero2) > 1e-4 :
    #     print('N0 problem ist {} nicht gleich {}'.format(valid_accuracy_without_zero, valid_accuracy_without_zero2))
    #     exit()

    #valid_accuracy_just_zero = (predictions[num_non_zero_targets:] == targets[num_non_zero_targets:]).sum().float() / targets[num_non_zero_targets:].size(0)
    valid_accuracy_just_zero = compare[zero_target].sum().float() / num_zero_targets

    # if torch.abs(valid_accuracy_just_zero - valid_accuracy_just_zero2) > 1e-4 :
    #     print('J0 problem ist {} nicht gleich {}'.format(valid_accuracy_just_zero, valid_accuracy_just_zero2))
    #     exit()

    return (valid_accuracy_without_zero, valid_accuracy_with_zero, valid_accuracy_just_zero, valid_accuracy_others_own)


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
               init_last=True, others_own=torch.zeros(1), others=torch.zeros(1), seq=-1):
    #print('fast adapt')
    flip_p = reid['ML']['flip_p']

    data, labels = batch
    data, labels = data, labels.to(device) + 1  # because 0 is for others class, transfer data to gpu
    ratio = torch.ones(ways + 1).to(device) * shots
    n = 1  # consider flip in indices
    if flip_p > 0.0:
        n = 2
        # get flipped feature maps, in combination with getitem method
        #data = data.view(-1, 256, 7, 7)
        #labels = labels.repeat_interleave(2)
        # because of flip
        #adaptation_indices = adaptation_indices.repeat_interleave(2)
        #evaluation_indices = evaluation_indices.repeat_interleave(2)

        # do all flipping here, put flips at the end
        #start_time = time.time()
        data = torch.cat((data, data.flip(-1)))
        labels = labels.repeat(2)
        ratio = torch.ones(ways + 1).to(device) * shots * 2
        #print("--- %s seconds --- [fa] flip " % (time.time() - start_time))

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways * n) * 2] = True  # true false true false ...
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    info = (sequence, ways, shots, iteration, train_task, taskID)

    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]



    # init last layer with the same weights
    if init_last:
        learner.init_last(ways)

    # plot loss and acc of val task before optimizing
    if train_task and iteration % 100 == 0 and True:
        predictions, _ = forward_pass_for_classifier_training(learner, adaptation_data,
                                                              adaptation_labels, ways, ratio=ratio,
                                                              train=True, return_scores=True)
        # train_accuracy = accuracy(train_predictions, adaptation_labels)
        predictions = predictions.argmax(dim=1).view(adaptation_labels.shape)
        t, counter = np.unique(predictions.cpu().numpy(), return_counts=True)
        print(f"[{seq}] check performance on train task train set BEFORE adaptation nach init")
        for i, c in enumerate(t):
            print(f"class {c} predicted {counter[i]} times")

    #shuffle = True
    #start_time = time.time()
    # if shuffle:
    #     idx = torch.randperm(adaptation_labels.nelement())
    #     adaptation_labels = adaptation_labels[idx]
    #     adaptation_data = adaptation_data[idx]
    #
    #     idx = torch.randperm(evaluation_labels.nelement())
    #     evaluation_labels = evaluation_labels[idx]
    #     evaluation_data = evaluation_data[idx]
        #print("--- %s seconds --- [fa] shuffle " % (time.time() - start_time))

    # get loss and acc of val set before optimizing
    # if train_task==False:
    #     predictions_before, validation_error_before = forward_pass_for_classifier_training(learner, evaluation_data,
    #                                                                          evaluation_labels, ways, return_scores=True)
    #     valid_accuracy_before = accuracy(predictions_before, evaluation_labels)


    # Adapt the model
    #start_time = time.time()
    for step in range(adaptation_steps):
        plot_inner = False
        use_others_inner_loop = False
        if (plotter != None) and (iteration % 10 == 0) and task == 0 and plot_inner:
            train_predictions, train_loss = forward_pass_for_classifier_training(learner, adaptation_data, adaptation_labels, ways, return_scores=True, ratio=ratio, train=True)
            train_accuracy = accuracy(train_predictions, adaptation_labels)
            #print(train_accuracy)
            #plotter.plot(epoch=step, loss=train_error, acc=train_accuracy, split_name='inner', info=info)
        elif use_others_inner_loop:
            num_others = len(others) + len(others_own) if len(others) > 1 else len(others_own)
            ID_others = 0
            ratio[ID_others] = num_others
            train_loss = forward_pass_for_classifier_training(learner, (adaptation_data, others_own, others),
                                                              torch.cat((adaptation_labels, torch.ones(num_others).long().to(device) * ID_others)),
                                                              ways, return_scores=False, ratio=ratio, train=False)
        else:
            train_predictions, train_loss = forward_pass_for_classifier_training(learner, adaptation_data,
                                                                                 adaptation_labels, ways, ratio=ratio, train=True, return_scores=True)
            #train_accuracy = accuracy(train_predictions, adaptation_labels)
        #graph = make_dot(train_loss, params=dict(learner.named_parameters()))
        #graph.format = 'png'
        #graph.render('graph2')
        learner.adapt(train_loss)  # Takes a gradient step on the loss and updates the cloned parameters in place
        train_loss = torch.zeros(1)

    #print(f"{time.time() - start_time:.5f} for inner loop {seq}")
    if train_task and iteration % 100 == 0 and True:
        predictions, _ = forward_pass_for_classifier_training(learner, adaptation_data,
                                                                              adaptation_labels, ways, ratio=ratio,
                                                                              train=True,return_scores=True)
        #train_accuracy = accuracy(train_predictions, adaptation_labels)
        predictions = predictions.argmax(dim=1).view(adaptation_labels.shape)
        t, counter = np.unique(predictions.cpu().numpy(), return_counts=True)
        print(f"[{seq}] check performance on train task train set AFTER adaptation")
        for i, c in enumerate(t):
            print(f"class {c} predicted {counter[i]} times")



    ID_others = 0
    if len(others) > 1 or len(others_own) > 1:
        #start_time = time.time()
        num_others = len(others) + len(others_own) if len(others) > 1 else len(others_own)
        ratio[ID_others] = num_others
        predictions, validation_loss = forward_pass_for_classifier_training(learner,  (evaluation_data, others_own, others), torch.cat((evaluation_labels,torch.ones(num_others).long().to(device) * ID_others)), ways,
                                                                 return_scores=True, ratio=ratio, train=False)

        valid_accuracy = accuracy(predictions,
                                  torch.cat((evaluation_labels, torch.ones(num_others).long().to(device) * ID_others)),
                                  iteration, train_task, seq, others_own.shape[0])
    else:
        predictions, validation_loss = forward_pass_for_classifier_training(learner,  evaluation_data, evaluation_labels, ways,
                                                                 return_scores=True, ratio=ratio, train=False, t=0.0)
        valid_accuracy = accuracy(predictions, evaluation_labels, iteration, train_task)

    return validation_loss, valid_accuracy

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


def sample_task(sets, nways, kshots, i_to_dataset, sample, val=False, num_tasks=-1, sample_uniform_DB=False):
    if sample_uniform_DB:
        if len(sets)>1:
            j = random.choice(range(7))  # sample which dataset, 0 MOT, 1,2,3 cuhk, 4,5,6 market to balance IDs
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
    # for debug
    if sample == False and val == False:
        i = 2
        seq = i_to_dataset[i]
    n = random.sample(nways, 1)[0]
    k = random.sample(kshots, 1)[0]
    transform = [l2l.data.transforms.FusedNWaysKShots(sets[i], n=n, k=k*2),
                 l2l.data.transforms.LoadData(sets[i]),
                 l2l.data.transforms.RemapLabels(sets[i], shuffle=True)]
    taskset = l2l.data.TaskDataset(dataset=sets[i],
                                   task_transforms=transform,
                                   num_tasks=num_tasks)
    try:
        batch = taskset.sample() # train
        return batch, n, k, seq, i
    except ValueError:
        return sample_task(sets, nways, kshots, i_to_dataset, sample, val, num_tasks)

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

    sampled_ids_train = {}

    # set all seeds
    torch.manual_seed(reid['seed'])
    torch.cuda.manual_seed(reid['seed'])
    np.random.seed(reid['seed'])
    torch.backends.cudnn.deterministic = True
    random.seed = (reid['seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            #datasets = [d for d in datasets if d not in ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']]
            datasets = [d for d in datasets if d not in ['cuhk03', 'market1501']]
            #datasets = [d for d in datasets if d == 'MOT17-02']

            for i,d in enumerate(datasets):
                i_to_dataset[i] = d
            print(f"Train with {datasets} and use {reid['dataloader']['validation_sequence']} as validation set")

            others_FM = []
            others_ID = []
            others_num = []
            others_seq_ID = []
            c = 0
            #data = {}
            bb_total = 0
            ids_total = 0
            for i, set in enumerate(datasets):
                seq = hf.get(set)
                d, l = seq.items()
                num = len(l[1])
                #data[set] = ((torch.tensor(l[1]), torch.tensor(d[1])))

                #ids, bb = statistics(data[set])
                #ids_total += ids
                #bb_total += bb
                # for i in data[set][0]:
                #     print(i)
                others_FM.append(torch.tensor(d[1]))
                others_ID.append(torch.tensor(l[1]))
                others_num.append((c, c+num))
                others_seq_ID.append(torch.ones(num)*i)
                c += num
                print(f"loaded {set} as seq id {i} with {num} samples")

            # print(
            #     'OVERALL: {} unique IDs, and {} BBs in total, av. {} BB/ID  '.format(ids_total, bb_total, (bb_total / ids_total)))
            others_FM = torch.cat(others_FM).to(device)  # contains all feature maps except val sequence
            others_ID = torch.cat(others_ID)
            others_seq_ID = torch.cat(others_seq_ID)
            print(f"Overall {others_FM.shape[0]} samples in others")


            if reid['dataloader']['validation_sequence'] != 'None':
                validation_data = hf.get(reid['dataloader']['validation_sequence'])
                d, l = validation_data.items()
                validation_FM = torch.tensor(d[1]).to(device)
                validation_ID = torch.tensor(l[1])
                print('loaded validation {}'.format(reid['dataloader']['validation_sequence']))
                print('Overall {} samples in val sequence'.format(validation_FM.shape[0]))
                #statistics(validation_data)
            print(f"--- {(time.time() - start_time_load):.4f} seconds --- for loading hdf5 database")

    start_time_tasks = time.time()

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
        validation_set = ML_dataset(fm=validation_FM, id=validation_ID)
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

    meta_datasets = []
    for i in others_num:
        meta_datasets.append(ML_dataset(
            fm=others_FM[i[0]:i[1]],
            id=others_ID[i[0]:i[1]],
        ))
    #meta_datasets = []
    #transforms = []
    #tasksets = []
    #tasksets = {}

    for i, s in enumerate(meta_datasets):
        meta_datasets[i] = l2l.data.MetaDataset(s)
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

    print(f"--- {(time.time() - start_time_tasks):.4f} seconds --- for construction of meta-datasets")
    print(f"--- {(time.time() - start_time):.4f} seconds --- for loading db and building meta-datasets ")
    

    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building CNN")

    obj_detect = FRCNN_FPN(num_classes=2).to(device)
    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                                          map_location=lambda storage, loc: storage))

    reID_network = reID_Model(obj_detect.roi_heads.box_head, obj_detect.roi_heads.box_predictor, n_list=nways_list)

    reID_network.train()
    reID_network.cuda()

    lr = float(reid['solver']['LR_init'])
    if reid['ML']['maml']:
        model = MAML(reID_network, lr=1e-3, first_order=True, allow_nograd=True)
        #opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        opt = torch.optim.Adam([
            {'params': model.module.head.parameters()},
            {'params': model.module.predictor.parameters()}
        ], lr=lr)
    if reid['ML']['learn_LR']:
        #predictor = obj_detect.roi_heads.box_predictor.cls_score
        predictor = torch.nn.Linear(in_features=1024, out_features=2).to(device)
        model = MetaSGD(reID_network, lr=1e-3, first_order=True, allow_nograd=True, global_LR=reid['ML']['global_LR'],
                        others_neuron_b=torch.nn.Parameter(predictor.bias[1].unsqueeze(0)),
                        others_neuron_w=torch.nn.Parameter(predictor.weight[1, :].unsqueeze(0)),
                        template_neuron_b=torch.nn.Parameter(predictor.bias[0].unsqueeze(0)),
                        template_neuron_w=torch.nn.Parameter(predictor.weight[0, :].unsqueeze(0)))

        lr_lr = float(reid['solver']['LR_LR'])
        opt = torch.optim.Adam([
            {'params': model.module.head.parameters()},
            {'params': model.template_neuron_bias},
            {'params': model.template_neuron_weight},
            {'params': model.others_neuron_bias},
            {'params': model.others_neuron_weight},
            {'params': model.lrs, 'lr': lr_lr}, # the template + others LR
            {'params': model.module.lrs[:4], 'lr': lr_lr} # LRs for the head
        ], lr=lr)


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
    ##################
    print("[*] Training ...")
    #print('use sqrt of ratio!')
    clamping_LR = True
    if clamping_LR:
        print('clamp LR < 0')

    loss_meta_train = AverageMeter('Loss', ':.4e')
    loss_meta_val = AverageMeter('Loss', ':.4e')
    acc_meta_train = AverageMeter('Acc', ':6.2f')
    acc_just_zero_meta_train = AverageMeter('Acc', ':6.2f')
    acc_all_meta_train = AverageMeter('Acc', ':6.2f')
    acc_others_own_meta_train = AverageMeter('Acc', ':6.2f')

    acc_meta_val = AverageMeter('Acc', ':6.2f')
    acc_just_zero_meta_val = AverageMeter('Acc', ':6.2f')
    acc_all_meta_val = AverageMeter('Acc', ':6.2f')
    acc_others_own_meta_val = AverageMeter('Acc', ':6.2f')

    # how to sample, uniform from the 3 db or uniform over sequences
    if ('market1501' and 'cuhk03') in datasets:
        sample_db = reid['ML']['db_sample_uniform']
    else:
        sample_db = False
        print(f"WORK without market und cuhk")
    # get others for market and cuhk
    #start_time = time.time()
    #others, others_id, others_val  = get_others(meta_datasets, -1, reid['ML']['num_others'], use_market=True)

    #others, others_id  = get_others(meta_datasets, -1, reid['ML']['num_others'], True, i_to_dataset)
    #others = l2l.data.UnionMetaDataset(meta_datasets)
    #start_time = time.time()
    #others_val = get_others(meta_datasets, -1, reid['ML']['num_others'], use_market=False)
    #logger.debug("--- %s seconds --- for get others" % (time.time() - start_time))
    #others_val = (others_val, others)

    # just use market for others
    # if reid['ML']['market_others']:
    #     others = get_others(meta_datasets, -1, reid['ML']['num_others'], use_market=True)
    #     ## split others in train and val, shuffle first
    #     idx = torch.randperm(others[1].nelement())
    #     others = (others[0][idx], others[1][idx])
    #     total_num_others = others[1].nelement()
    #     output, inv_idx, counts = torch.unique(others[1], return_inverse=True, return_counts=True)
    #     num_val_id = int(output.nelement() * 0.3)
    #     num_samples = 0
    #     others_ = torch.tensor([]).to(device)
    #     for i in range(num_val_id):
    #         num_samples += counts[i]
    #         others_ = torch.cat((others_, (others[0][(others[1] == output[i])]).float()))
    #     others_val = (others_, torch.ones(others_.shape[0]).long().to(device))
    #     others_ = torch.tensor([]).to(device)
    #     for i in range(num_val_id, output.nelement()):
    #         others_ = torch.cat((others_, (others[0][(others[1] == output[i])]).float()))
    #     others = (others_, torch.ones(others_.shape[0]).long().to(device))
    #     print('work with {} train samples and {} val samples for others'.format(total_num_others-num_samples, num_samples))

    plotter = None
    if reid['solver']["plot_training_curves"]:
        now = datetime.datetime.now()
        run_name = now.strftime("%Y-%m-%d_%H:%M_") + reid['name']
        plotter = VisdomLinePlotter_ML(env=run_name, offline=reid['solver']['plot_offline'],
                                       info=(nways_list, kshots_list, meta_batch_size, num_tasks,
                                             num_tasks_val, reid['dataloader']['validation_sequence'],
                                             reid['ML']['flip_p']))

    # safe model 10 times
    safe_every = int(_config['reid']['solver']['iterations'] / 10)
    init_last = reid['ML']['init_last']

    for iteration in range(1, _config['reid']['solver']['iterations']+1):
        # debug
        #if iteration % 50 or True:
        if False:
            print(f"others neuron bias {model.others_neuron_bias.item()}")
            print(f"others neuron weight mean {model.others_neuron_weight.mean().item()}")
            print(f"others neuron weightLR mean {model.lrs[0].mean().item()}")
            print(f"others neuron biasLR mean {model.lrs[1].item()}")

            print(f"template neuron bias {model.template_neuron_bias.item()}")
            print(f"template neuron weight mean {model.template_neuron_weight.mean().item()}")
            print(f"template neuron weightLR mean {model.lrs[2].mean().item()}")
            print(f"template neuron biasLR mean {model.lrs[3].item()}")

            print(f"head neuron bias {model.module.head.fc7.bias.mean().item()}")
            print(f"head neuron weight mean {model.module.head.fc7.weight.mean().item()}")
            print(f"head neuron weightLR mean {model.module.lrs[2].mean().item()}")
            print(f"head neuron biasLR mean {model.module.lrs[3].mean().item()}")


        start_time_iteration = time.time()
        opt.zero_grad()

        # performance of train task
        meta_train_loss = 0.0
        meta_train_accuracy = 0.0
        # with zero
        meta_train_accuracy_all = 0.0
        # just zero
        meta_train_accuracy_just_zero = 0.0
        meta_train_accuracy_others_own = 0.0

        # performance of val task
        # without zero class , matches meta_train_error
        meta_valid_loss = 0.0
        meta_valid_accuracy = 0.0
        # with zero
        meta_valid_accuracy_all = 0.0
        # just zero
        meta_valid_accuracy_just_zero = 0.0
        meta_valid_accuracy_others_own = 0.0

        # 1 validation task per meta batch size train tasks
        # Compute meta-validation loss - validation sequence
        # here no backward in outer loop, just inner
        if len(validation_set) > 0:
            learner = model.clone()
            batch, nways, kshots, _, taskID = sample_task(validation_set, nways_list, kshots_list, i_to_dataset,
                                                          reid['ML']['sample_from_all'], val=True,
                                                          num_tasks=num_tasks_val)

            # get others from current seq which are not used for task
            batch, used_labels = batch
            others_val_idx = [
                idx for l, idx in validation_set[0].labels_to_indices.items()
                if l not in used_labels]
            others_val_idx = functools.reduce(operator.iconcat, others_val_idx, [])
            others_val = validation_set[0][others_val_idx][0]

            # debug = [idx for l, idx in validation_set[0].labels_to_indices.items()
            #          if l in used_labels]
            # debug = functools.reduce(operator.iconcat, debug, [])
            #
            # if iteration % 10 == 0:
            #     print('val task used labels {}'.format(used_labels))
            #     print('{} ind for tasks'.format(len(debug), debug))
            #     print('{} ind for others from seq'.format(len(others_val_idx),others_val_idx))

            # if (nways, kshots) not in sampled_val.keys():
            #     sampled_val[(nways, kshots)] = 1
            # else:
            #     sampled_val[(nways, kshots)] += 1
            start_time = time.time()
            evaluation_loss, evaluation_accuracy = fast_adapt(batch,
                                                              learner,
                                                              adaptation_steps,
                                                              kshots,
                                                              nways,
                                                              device,
                                                              False,
                                                              plotter,
                                                              iteration,
                                                              reid['dataloader']['validation_sequence'],
                                                              -1,
                                                              taskID,
                                                              reid,
                                                              init_last,
                                                              others_val,
                                                              others_FM
                                                              )
            #print(f"{time.time() - start_time:.5f} for fast adapt val")
            evaluation_accuracy, evaluation_accuracy_all, evaluation_accuracy_just_zero, evaluation_accuracy_others_own = evaluation_accuracy

            meta_valid_loss += evaluation_loss.item()
            meta_valid_accuracy += evaluation_accuracy.item()  # accuracy without zero class
            meta_valid_accuracy_all += evaluation_accuracy_all.item()  # accuracy with zeros class
            meta_valid_accuracy_just_zero += evaluation_accuracy_just_zero.item()  # track acc just zero class
            meta_valid_accuracy_others_own += evaluation_accuracy_others_own.item()  # track acc just zero class

            loss_meta_val.update(evaluation_loss.item())
            acc_meta_val.update(evaluation_accuracy.item())
            acc_just_zero_meta_val.update(evaluation_accuracy_just_zero.item())  # others from own sequence
            acc_all_meta_val.update(evaluation_accuracy_all.item())  # others from others sequences
            acc_others_own_meta_val.update(evaluation_accuracy_others_own.item())  # others from OWN sequence

        for task in range(meta_batch_size):
            # Compute meta-training loss
            #start_time=time.time()
            learner = model.clone()  # back-propagating losses on the cloned module will populate the buffers of the original module
            batch, nways, kshots, sequence, taskID = sample_task(meta_datasets, nways_list, kshots_list, i_to_dataset,
                                                                 reid['ML']['sample_from_all'], num_tasks=num_tasks,
                                                                 sample_uniform_DB=sample_db)
            sequence, sequence_idx = sequence
            batch, used_labels = batch
            others_val_idx = [
                idx for l, idx in meta_datasets[sequence_idx].labels_to_indices.items()
                if l not in used_labels]
            others_val_idx = functools.reduce(operator.iconcat, others_val_idx, [])
            others_val = meta_datasets[sequence_idx][others_val_idx][0]

            if sequence not in sampled_ids_train.keys():
                sampled_ids_train[sequence] = {}

            # for u in used_labels:
            #     if u not in sampled_ids_train[sequence].keys():
            #         sampled_ids_train[sequence][u] = 1
            #     else:
            #         sampled_ids_train[sequence][u] += 1

            # debug = [idx for l, idx in meta_datasets[sequence_idx].labels_to_indices.items()
            #          if l in used_labels]
            # debug = functools.reduce(operator.iconcat, debug, [])
            #
            # if iteration % 10 == 0:
            #     print('train task used labels {}'.format(used_labels))
            #     print('{} ind for tasks'.format(len(debug), debug))
            #     print('{} ind for others from seq {}'.format(len(others_val_idx), sequence))
            #     #print(others_val_idx)
            #     print('len von others ohne seq {}'.format(others_FM[others_seq_ID!=sequence_idx].shape[0]))

            if sequence not in sampled_train.keys():
                sampled_train[sequence] = {}

            if (nways, kshots) not in sampled_train[sequence].keys():
                sampled_train[sequence][(nways, kshots)] = 1
            else:
                sampled_train[sequence][(nways, kshots)] += 1
            start_time = time.time()
            evaluation_loss, evaluation_accuracy = fast_adapt(batch,
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
                                                               others_val,
                                                               others_FM[others_seq_ID != sequence_idx],
                                                               seq=sequence
                                                               )

            #graph = make_dot(evaluation_loss, params=dict(learner.named_parameters()))
            #graph.format = 'png'
            #graph.render('graph3')
            evaluation_loss.backward()  # compute gradients, populate grad buffers of maml
            #print(f"{time.time() - start_time:.5f} for fast adapt train + backward")

            evaluation_accuracy, evaluation_accuracy_all, evaluation_accuracy_just_zero, evaluation_accuracy_others_own = evaluation_accuracy

            meta_train_loss += evaluation_loss.item()
            meta_train_accuracy += evaluation_accuracy.item()
            meta_train_accuracy_all += evaluation_accuracy_all.item()  # accuracy with zeros class
            meta_train_accuracy_just_zero += evaluation_accuracy_just_zero.item()  # track acc just zero class
            meta_train_accuracy_others_own += evaluation_accuracy_others_own.item()  # track acc others OWN

            loss_meta_train.update(evaluation_loss.item())
            acc_meta_train.update(evaluation_accuracy.item())
            acc_just_zero_meta_train.update(evaluation_accuracy_just_zero.item())  # others from own sequence
            acc_all_meta_train.update(evaluation_accuracy_all.item())  # others from others sequences
            acc_others_own_meta_train.update(evaluation_accuracy_others_own.item())  # others from OWN sequence

            evaluation_loss = torch.zeros(1)

        # update the best value for loss
        if len(validation_set) > 0:
            safe_best_loss = loss_meta_val.update_best(loss_meta_val.avg, iteration)
        else:
            safe_best_loss = loss_meta_val.update_best(loss_meta_train.avg, iteration)

        if iteration % 10 == 0:
            # Print some metrics
            print(f"\nIteration {iteration} ,average over meta batch size {meta_batch_size}")
            print(f"{'Train Loss':<50} {meta_train_loss / meta_batch_size:.8f}")
            print(f"{'Train Accuracy task':<50} {meta_train_accuracy / meta_batch_size:.6f}")
            print(f"{'Train Accuracy task+others':<50} {meta_train_accuracy_all / meta_batch_size:.6f}")
            print(f"{'Train Accuracy just others':<50} {meta_train_accuracy_just_zero / meta_batch_size:.6f}")
            print(f"{'Train Accuracy others own sequence':<50} {meta_train_accuracy_others_own / meta_batch_size:.6f}")

            # print(f"{'Valid Loss':<50} {meta_valid_loss / meta_batch_size}")
            # print(f"{'Valid Accuracy task':<50} {meta_valid_accuracy / meta_batch_size}")
            # print(f"{'Valid Accuracy task+others':<50} {meta_valid_accuracy_all / meta_batch_size}")
            # print(f"{'Valid Accuracy just others:':<50} {meta_valid_accuracy_just_zero / meta_batch_size}")
            # print(f"{'Valid Accuracy others own sequence':<50} {meta_valid_accuracy_others_own / meta_batch_size}")

            print(f"{'Valid Loss':<50} {meta_valid_loss:.8f}")
            print(f"{'Valid Accuracy task':<50} {meta_valid_accuracy:.6f}")
            print(f"{'Valid Accuracy task+others':<50} {meta_valid_accuracy_all:.6f}")
            print(f"{'Valid Accuracy just others:':<50} {meta_valid_accuracy_just_zero:.6f}")
            print(f"{'Valid Accuracy others own sequence':<50} {meta_valid_accuracy_others_own:.6f}")

            print(f"\n{'Mean Train Loss':<50} {loss_meta_train.avg:.8f}")
            print(f"{'Mean Train Accuracy task':<50} {acc_meta_train.avg:.6f}")
            print(f"{'Mean Train Accuracy task+others':<50} {acc_all_meta_train.avg:.6f}")
            print(f"{'Mean Train Accuracy just others':<50} {acc_just_zero_meta_train.avg:.6f}")
            print(f"{'Mean Train Accuracy others own sequence':<50} {acc_others_own_meta_train.avg:.6f}")

            print(f"{'Mean Val Loss':<50} {loss_meta_val.avg:.8f}")
            print(f"{'Mean Val Accuracy task':<50} {acc_meta_val.avg:.6f}")
            print(f"{'Mean Val Accuracy task+others':<50} {acc_all_meta_val.avg:.6f}")
            print(f"{'Mean Val Accuracy just others':<50} {acc_just_zero_meta_val.avg:.6f}")
            print(f"{'Mean Val Accuracy others own sequence':<50} {acc_others_own_meta_val.avg:.6f}")

            if reid['ML']['learn_LR'] and reid['ML']['global_LR']:
                for p in model.lrs:
                    print('LR: {}'.format(p.item()))

        #if iteration>30000 and (iteration%safe_every==0 or safe_best_loss==1) and (iteration%100==0):
        if iteration > 1000 and (iteration%safe_every == 0 or safe_best_loss == 1):
            if iteration % 10 == 0:
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

        if reid['solver']["plot_training_curves"] and (iteration % 100 == 0 or iteration == 1):
            if reid['ML']['learn_LR'] and reid['ML']['global_LR'] :
                plotter.plot(epoch=iteration, loss=meta_train_loss/meta_batch_size,
                             acc=meta_train_accuracy/meta_batch_size, split_name='train_task_val_set', LR=model.lrs[0])
            else:
                plotter.plot(epoch=iteration, loss=meta_train_loss / meta_batch_size,
                             acc=meta_train_accuracy / meta_batch_size, split_name='train_task_val_set')
            plotter.plot(epoch=iteration, loss=loss_meta_train.avg, acc=acc_meta_train.avg, split_name='train_task_val_set MEAN')
            plotter.plot(epoch=iteration, loss=meta_valid_loss, acc=meta_valid_accuracy, split_name='val_task_val_set')
            plotter.plot(epoch=iteration, loss=loss_meta_val.avg, acc=acc_meta_val.avg, split_name='val_task_val_set MEAN')
            plotter.plot(epoch=iteration, loss=-1, acc=acc_others_own_meta_train.avg, split_name='train_task_val_set_others_own MEAN')
            plotter.plot(epoch=iteration, loss=-1, acc=acc_all_meta_train.avg, split_name='train_task_val_set_all MEAN')
            plotter.plot(epoch=iteration, loss=-1, acc=acc_others_own_meta_val.avg, split_name='val_task_val_set_others_own MEAN')
            plotter.plot(epoch=iteration, loss=-1, acc=acc_all_meta_val.avg, split_name='val_task_val_set_all MEAN')
            #plotter.plot(epoch=iteration, loss=-1, acc=acc_just_zero_meta_train.avg, split_name='train_task_val_set_just_zero MEAN')
            #plotter.plot(epoch=iteration, loss=-1, acc=acc_just_zero_meta_val.avg, split_name='val_task_val_set_just_zero MEAN')



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
                if (lr < 0).sum().item() > 0:
                    lr_mask = (lr > 0)
                    model.lrs[i] = torch.nn.Parameter(lr*lr_mask)

            for i, lr in enumerate(model.module.lrs):
                if (lr < 0).sum().item() > 0:
                    lr_mask = (lr > 0)
                    model.module.lrs[i] = torch.nn.Parameter(lr*lr_mask)
        #print('OVERALL: 1 iteration in {}'.format(time.time()-start_time_iteration))

        if iteration % 100 == 0:
            print('sampled tasks from train {}'.format(sampled_train))
            print('sampled tasks from val {}'.format(sampled_val))
            #print('\n check sampled IDs per dataset {}'.format(sampled_ids_train))

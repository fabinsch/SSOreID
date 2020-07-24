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

import torch
import torch.nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
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
import h5py
import datetime

ex = Experiment()
ex.add_config('experiments/cfgs/ML_reid.yaml')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetaSGD_(l2l.algorithms.MetaSGD):
    def __init__(self, model, lr=1.0, first_order=False, lrs=None):
        super(l2l.algorithms.MetaSGD, self).__init__()
        self.module = model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if lrs is None:
            lrs = [torch.ones(1).to(device) * lr for p in model.parameters()]
            lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
        self.lrs = lrs
        self.first_order = first_order

class ML_dataset(Dataset):
    def __init__(self, data, flip_p):
        self.data = data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            features = self.data[1][item].flip(-1).unsqueeze(0)
            features = torch.cat((self.data[1][item].unsqueeze(0), features))

            label = self.data[0][item] #.unsqueeze(0)
            #label = torch.cat((self.data[0][item].unsqueeze(0), label))
            return (features, label)

    def __len__(self):
        return len(self.data[0])


class reID_Model(torch.nn.Module):
    def __init__(self, head, predictor, n):
        super(reID_Model, self).__init__()
        self.head = head
        self.predictor = predictor.cls_score
        self.num_output = n
        self.additional_layer = {}
        if len(n)>0:
            for i in n:
                if i>2:
                    self.additional_layer[i] = torch.nn.Linear(1024, i - 2)
                else:
                    self.additional_layer[i] = None
        else:
            self.additional_layer = None

    def forward(self, x, nways):
        feat = self.head(x)
        x = self.predictor(feat)
        if self.additional_layer != None and nways>2:
            self.additional_layer[nways].to(device)
            add = self.additional_layer[nways](feat)
            x = torch.cat((x, add), dim=1)
        return x


def forward_pass_for_classifier_training(learner, features, scores, nways, eval=False, return_scores=False):

    if eval:
        learner.eval()
    class_logits = learner(features, nways)

    if return_scores:
        pred_scores = F.softmax(class_logits, -1)
        loss = F.cross_entropy(class_logits, scores.long())
        if eval:
            learner.train()
        return pred_scores.detach(), loss

    loss = F.cross_entropy(class_logits, scores.long())

    if eval:
        learner.train()
    else:
        return loss


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

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

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def fast_adapt(batch, learner, adaptation_steps, shots, ways,  device, train_task=True,
               plotter=None, iteration=-1, sequence=None, task=-1, taskID=-1):
    valid_accuracy_before = torch.zeros(1)
    validation_error_before = torch.zeros(1)

    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    # get flipped feature maps
    data = data.view(-1, 256, 7, 7)
    labels = labels.repeat_interleave(2)

    info = (sequence, ways, shots, iteration, train_task, taskID)

    # because of flip
    adaptation_indices = adaptation_indices.repeat_interleave(2)
    evaluation_indices = evaluation_indices.repeat_interleave(2)

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
    if shuffle==True:
        idx = torch.randperm(adaptation_labels.nelement())
        adaptation_labels = adaptation_labels[idx]
        adaptation_data = adaptation_data[idx]

        idx = torch.randperm(evaluation_labels.nelement())
        evaluation_labels = evaluation_labels[idx]
        evaluation_data = evaluation_data[idx]

    # if train_task==False:
    #     predictions_before, validation_error_before = forward_pass_for_classifier_training(learner, evaluation_data,
    #                                                                          evaluation_labels, ways, return_scores=True)
    #     valid_accuracy_before = accuracy(predictions_before, evaluation_labels)

    # Adapt the model
    for step in range(adaptation_steps):
        if (plotter != None) and (iteration%500==0) and task%50==0:
            train_predictions, train_error = forward_pass_for_classifier_training(learner, adaptation_data, adaptation_labels, ways, return_scores=True)
            train_accuracy = accuracy(train_predictions, adaptation_labels)
            plotter.plot(epoch=step, loss=train_error, acc=train_accuracy, split_name='inner', info=info)
        else:
            train_error = forward_pass_for_classifier_training(learner, adaptation_data, adaptation_labels, ways)
        #train_error /= len(adaptation_data)
        #learner.adapt(train_error, first_order=True)  # for meta sgd specify first order here
        learner.adapt(train_error)  # Takes a gradient step on the loss and updates the cloned parameters in place


    # Evaluate the adapted model
    predictions, validation_error = forward_pass_for_classifier_training(learner, evaluation_data, evaluation_labels, ways, return_scores=True)
    valid_accuracy = accuracy(predictions, evaluation_labels)

    if train_task==False:
        valid_accuracy = (valid_accuracy, valid_accuracy_before)
        validation_error = (validation_error, validation_error_before)

    return validation_error, valid_accuracy

def statistics(dataset):
    unique_id, counter = np.unique(dataset[0].numpy(), return_counts=True)
    num = len(unique_id)
    samples_per_id, counter_samples = np.unique(counter, return_counts=True)
    print('in total {} unique IDs, print until 80 samples per ID'.format(num))
    for i in range(len(counter_samples)):
        if samples_per_id[i]<80:
            print('{} samples per ID: {} times'.format(samples_per_id[i], counter_samples[i]))

def sample_task(tasksets, i_to_dataset, sample, val=False):
    try:
        i = random.choice(range(len(tasksets)))  # sample sequence
        seq = i_to_dataset[i]
        seq_tasks = tasksets[list(tasksets.keys())[i]]
        if sample==False and val==False:
            i=1
            seq_tasks = tasksets[list(tasksets.keys())[i]]
            seq = i_to_dataset[i]
        j = random.choice(range(len(seq_tasks)))  # sample task
        nways = seq_tasks[j].task_transforms[0].n
        kshots = int(seq_tasks[j].task_transforms[0].k / 2)
        batch = seq_tasks[j].sample()
        return batch, nways, kshots, seq, j
    except ValueError:
        nways = seq_tasks[j].task_transforms[0].n
        kshots = int(seq_tasks[j].task_transforms[0].k / 2)
        # if len(tasksets)>1:
        #     print('Problem to sample {} ways and {} shots from {}'.format(nways, kshots, i_to_dataset[i]))
        # else:
        #     print('Problem to sample {} ways and {} shots from validation sequence'.format(nways, kshots))
        return sample_task(tasksets, i_to_dataset, sample, val)
        # i = random.choice(range(len(tasksets)))
        # batch = tasksets[i].sample()

@ex.automain
def my_main(_config, reid, _run):
    sacred.commands.print_config(_run)

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
            for  i,d in enumerate(datasets):
                i_to_dataset[i] = d
            print('Train with {} and use {} as validation set'.format(datasets, reid['dataloader']['validation_sequence']))

            data = {}
            for set in datasets:
                seq = hf.get(set)
                d, l = seq.items()
                data[set] = ((torch.tensor(l[1]), torch.tensor(d[1])))
                print('loaded {}'.format(set))
                #statistics(data[set])
                # for i in data[set][0]:
                #     print(i)

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


    validation_set = ML_dataset(validation_data, reid['ML']['flip_p'])
    validation_set = l2l.data.MetaDataset(validation_set)
    val_transform = []
    for i in nways_list:
        for kshots in kshots_list:
            val_transform.append([l2l.data.transforms.FusedNWaysKShots(validation_set, n=i, k=kshots * 2),
                                  l2l.data.transforms.LoadData(validation_set),
                                  l2l.data.transforms.RemapLabels(validation_set, shuffle=True)
                                  ])
    # val_transform =[l2l.data.transforms.FusedNWaysKShots(validation_set, n=nways, k=kshots * 2),
    #      l2l.data.transforms.LoadData(validation_set),
    #      l2l.data.transforms.RemapLabels(validation_set, shuffle=True)
    #      ]
    val_taskset = {}
    for t in val_transform:
        if 'val' not in val_taskset.keys():
            val_taskset['val']=[l2l.data.TaskDataset(dataset=validation_set,
                                                    task_transforms=t,
                                                    num_tasks=num_tasks_val)]
        else:
            val_taskset['val'].append(l2l.data.TaskDataset(dataset=validation_set,
                                                 task_transforms=t,
                                                 num_tasks=num_tasks_val))

    # val_taskset = l2l.data.TaskDataset(dataset=validation_set,
    #                                      task_transforms=val_transform,
    #                                      num_tasks=num_tasks_val)

    sequences = []
    for dataset in data:
        sequences.append(ML_dataset(data[dataset], reid['ML']['flip_p']))

    meta_datasets = []
    #transforms = []
    #tasksets = []
    tasksets = {}

    for i, s in enumerate(sequences):
        meta_datasets = l2l.data.MetaDataset(s)
        for j in nways_list:
            for kshots in kshots_list:
                transform = [l2l.data.transforms.FusedNWaysKShots(meta_datasets, n=j, k=kshots*2),
                    l2l.data.transforms.LoadData(meta_datasets),
                     l2l.data.transforms.RemapLabels(meta_datasets, shuffle=True)]
                if i not in tasksets.keys():
                    tasksets[i] = [l2l.data.TaskDataset(dataset=meta_datasets,
                                                     task_transforms=transform,
                                                     num_tasks=num_tasks)]
                else:
                    tasksets[i].append(l2l.data.TaskDataset(dataset=meta_datasets,
                                                     task_transforms=transform,
                                                     num_tasks=num_tasks))
    print("--- %s seconds --- for construction of meta-datasets and tasks" % (time.time() - start_time_tasks))
    print("--- %s seconds --- for loading db and building tasksets " % (time.time() - start_time))
    
    ##########################
    # Clean #
    ##########################
    del data
    del validation_data
    del meta_datasets
    del validation_set
    del sequences
    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building CNN")

    obj_detect = FRCNN_FPN(num_classes=2).to(device)
    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                                          map_location=lambda storage, loc: storage))

    box_head_classification = obj_detect.roi_heads.box_head
    box_predictor_classification = obj_detect.roi_heads.box_predictor

    reID_network = reID_Model(box_head_classification, box_predictor_classification, n=nways_list)
    reID_network.train()
    reID_network.cuda()

    for n in nways_list:
        if n>2:
            for p in reID_network.additional_layer[n].parameters():
                p.requires_grad = False


    maml = l2l.algorithms.MAML(reID_network, lr=1e-3, first_order=True, allow_nograd=True)
    opt = torch.optim.Adam(maml.parameters(), lr=1e-4)

    # for p in maml.parameters():
    #     a = p.numel
    # opt = torch.optim.Adam(maml.parameters(), lr=4e-3)
    # print(sum(p.numel() for p in box_head.parameters() if p.requires_grad))

    #meta_sgd = MetaSGD_(reID_network, lr=1e-3)

    ##################
    # Begin training #
    ##################
    print("[*] Training ...")
    losses_meta_train = AverageMeter('Loss', ':.4e')
    losses_meta_val = AverageMeter('Loss', ':.4e')
    acc_meta_train = AverageMeter('Acc', ':6.2f')
    acc_meta_val = AverageMeter('Acc', ':6.2f')

    plotter=None
    if reid['solver']["plot_training_curves"]:
        now = datetime.datetime.now()
        run_name = now.strftime("%Y-%m-%d_%H:%M")
        plotter = VisdomLinePlotter_ML(env=run_name, offline=reid['solver']['plot_offline'],
                                       info=(nways_list, kshots_list, meta_batch_size, num_tasks,
                                             num_tasks_val, reid['dataloader']['validation_sequence'],
                                             reid['ML']['flip_p']))

    for iteration in range(_config['reid']['solver']['iterations']):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_valid_error_before = 0.0
        meta_valid_accuracy_before = 0.0
        for task in range(meta_batch_size):
            # Compute meta-validation loss
            # here no backward in outer loop, just inner
            learner = maml.clone()
            #learner = meta_sgd.clone()
            batch, nways, kshots, _, taskID = sample_task(val_taskset, i_to_dataset, reid['ML']['sample_from_all'], val=True)

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
                                                               taskID)

            evaluation_error, evaluation_error_before = evaluation_error
            evaluation_accuracy, evaluation_accuracy_before = evaluation_accuracy
            meta_valid_error_before += evaluation_error_before.item()
            meta_valid_accuracy_before += evaluation_accuracy_before.item()

            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            losses_meta_val.update(evaluation_error.item())
            acc_meta_val.update(evaluation_accuracy.item())


            # Compute meta-training loss
            learner = maml.clone()  #back-propagating losses on the cloned module will populate the buffers of the original module
            #learner = meta_sgd.clone()  #back-propagating losses on the cloned module will populate the buffers of the original module

            batch, nways, kshots, sequence, taskID = sample_task(tasksets, i_to_dataset, reid['ML']['sample_from_all'])
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
                                                               taskID)
            evaluation_error.backward()  # compute gradients, populate grad buffers of maml
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            losses_meta_train.update(evaluation_error.item())
            acc_meta_train.update(evaluation_accuracy.item())

        if iteration%100==0:
            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
            print('\n')
            print('Mean Meta Train Error', losses_meta_train.avg)
            print('Mean Meta Train Accuracy', acc_meta_train.avg)
            print('Mean Meta Val Error', losses_meta_val.avg)
            print('Mean Meta Val Accuracy', acc_meta_val.avg)
            # print('safe state dict')
            # model = osp.join(output_dir, 'reID_Network.pth')
            # torch.save(maml.state_dict(), model)

        if iteration%100==0:
            model_s = '{}_reID_Network.pth'.format(iteration)
            print('safe state dict {}'.format(model_s))
            model = osp.join(output_dir, model_s)
            torch.save(maml.state_dict(), model)



        if reid['solver']["plot_training_curves"]:
            plotter.plot(epoch=iteration, loss=meta_train_error/meta_batch_size, acc=meta_train_accuracy/meta_batch_size, split_name='train_task_val_set')
            plotter.plot(epoch=iteration, loss=losses_meta_train.avg, acc=acc_meta_train.avg, split_name='train_task_val_set MEAN')
            plotter.plot(epoch=iteration, loss=meta_valid_error/meta_batch_size, acc=meta_valid_accuracy/meta_batch_size, split_name='val_task_val_set')
            plotter.plot(epoch=iteration, loss=losses_meta_val.avg, acc=acc_meta_val.avg, split_name='val_task_val_set MEAN')
            #plotter.plot(epoch=iteration, loss=meta_valid_error_before/meta_batch_size, acc=meta_valid_accuracy_before/meta_batch_size, split_name='val_task_val_set_before')
        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        # for p in meta_sgd.parameters():
        #     p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    # meta_test_error = 0.0
    # meta_test_accuracy = 0.0
    # for task in range(meta_batch_size):
    #     # Compute meta-testing loss
    #     learner = maml.clone()
    #     batch = tasksets.test.sample()
    #     evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                        learner,
    #                                                        loss,
    #                                                        adaptation_steps,
    #                                                        shots,
    #                                                        ways,
    #                                                        device)
    #     meta_test_error += evaluation_error.item()
    #     meta_test_accuracy += evaluation_accuracy.item()
    # print('Meta Test Error', meta_test_error / meta_batch_size)
    # print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)

    #
    #

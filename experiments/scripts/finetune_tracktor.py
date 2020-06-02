import os
import time
from os import path as osp
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import motmetrics as mm

mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import random
import sacred
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')
ex.add_named_config('cfg_classification', 'experiments/cfgs/cfg_classification.yaml')
ex.add_named_config('SLURMcfg_classification', 'experiments/cfgs/SLURMcfg_classification.yaml')

########### DEFAULT################

# hacky workaround to load the corresponding configs and not having to har
# dcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ex.automain
def main(tracktor, reid, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True
    random.seed=(tracktor['seed'])

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'], tracktor['output_subdir'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector.")

    reid_network = None

    if tracktor['tracker']['reid_siamese']:
        # reid
        reid_network = resnet50(pretrained=False, **reid['cnn']).to(device)
        reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                                map_location=lambda storage, loc: storage))
        reid_network.eval()

    time_total = 0
    num_frames = 0
    mot_accums = []
    dataset = Datasets(tracktor['dataset'])

    for seq in dataset:

        obj_detect = FRCNN_FPN(num_classes=2).to(device)
        obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                                              map_location=lambda storage, loc: storage))
        obj_detect.eval()

        # tracktor
        if 'oracle' in tracktor:
            tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
        else:
            tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

        start = time.time()

        _log.info(f"Tracking: {seq}")

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        for i, frame in enumerate(tqdm(data_loader)):
            if i>=0 and len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                tracker.step(frame, i)
                num_frames += 1

        results = tracker.get_results()

        time_total += time.time() - start

        _log.info(f"Tracks found: {len(results)}")
        _log.info(f"Runtime for {seq}: {time.time() - start :.1f} s.")
        _log.info(f"Total number of REIDs: {tracker.num_reids}")
        _log.info(f"Total number of Trainings: {tracker.num_training}")
        _log.info(f"Number of skipped samples because of IoU restriction: {tracker.c_skipped_for_train_iou}")
        _log.info(f"Number of skipped samples because of IoU restriction (with just one active frame): {tracker.c_skipped_and_just_and_frame_active}")
        #_log.info(f"It was trained on: {tracker.train_on}")
        _log.info(f"It happen x times that it was killed and reid in same step: {tracker.count_killed_this_step_reid}")
        _log.info(f"Number of tracks which are just active 1 frame: {tracker.c_just_one_frame_active}")
        _log.info(f"Epochs trained: {tracker.trained_epochs}")
        _log.info(f"Epochs average: {sum(tracker.trained_epochs) / len(tracker.trained_epochs)}")

        if tracktor['interpolate']:
            results = interpolate(results)

        if seq.no_gt:
            _log.info(f"No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        _log.info(f"Writing predictions to: {output_dir}")

        seq.write_results(results, output_dir)
        if tracktor['write_images']:
            plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

        del tracker
        del obj_detect
        torch.cuda.empty_cache()


    _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
          f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")

    if mot_accums:
        summary = evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)
        summary.to_pickle("output/finetuning_results/results_{}_{}_{}_{}_{}.pkl".format(tracktor['output_subdir'],
                                                                                        tracktor['tracker']['finetuning'][
                                                                                            'max_displacement'],
                                                                                        tracktor['tracker']['finetuning'][
                                                                                            'batch_size'],
                                                                                        tracktor['tracker']['finetuning'][
                                                                                            'learning_rate'],
                                                                                        tracktor['tracker']['finetuning'][
                                                                                            'epochs']))

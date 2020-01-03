import os
import time
from os import path as osp

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
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
ex.add_named_config('cfg1', 'experiments/cfgs/hp_search/cfg1.yaml')
ex.add_named_config('cfg_caro_ws1', 'experiments/cfgs/hp_search/cfg_caro_ws1.yaml')
ex.add_named_config('cfg_caro_ws2', 'experiments/cfgs/hp_search/cfg_caro_ws2.yaml')
ex.add_named_config('cfg_caro_ws3', 'experiments/cfgs/hp_search/cfg_caro_ws3.yaml')
ex.add_named_config('cfg_caro_local', 'experiments/cfgs/hp_search/cfg_caro_local.yaml')


########### DEFAULT################

# hacky workaround to load the corresponding configs and not having to har
# dcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


def compare_models(m1, m2):
    for key_item_1, key_item_2 in zip(m1.state_dict().items(), m2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            continue
        else:
            return False
    return True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@ex.automain
def main(tracktor, reid, _config, _log, _run):

    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

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

    obj_detect = FRCNN_FPN(num_classes=2).to(device)
    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                                          map_location=lambda storage, loc: storage))

    obj_detect.eval()

    #obj_detect_copy = FRCNN_FPN(num_classes=2)
    #obj_detect_copy.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
    #                                map_location=lambda storage, loc: storage))
    #obj_detect_copy.eval()

    # reid
    reid_network = resnet50(pretrained=False, **reid['cnn']).to(device)
    reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                            map_location=lambda storage, loc: storage))
    reid_network.eval()

    #reid_network_copy = resnet50(pretrained=False, **reid['cnn'])
    #reid_network_copy.load_state_dict(torch.load(tracktor['reid_weights'],
    #                                        map_location=lambda storage, loc: storage))
    #reid_network_copy.eval()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    time_total = 0
    num_frames = 0
    mot_accums = []
    dataset = Datasets(tracktor['dataset'])

    for seq in dataset:
         #if not "04" in str(seq) and not "02" in str(seq):
         #   print("Skipping")
         #   continue

         #print("Reid network not changed? {}".format(compare_models(reid_network, tracker.reid_ne04twork)))
         #print("Object detection network not changed? {}".format(compare_models(obj_detect_copy, tracker.obj_detect)))

         tracker.reset()

         start = time.time()

         _log.info(f"Tracking: {seq}")

         data_loader = DataLoader(seq, batch_size=1, shuffle=False)
         for i, frame in enumerate(tqdm(data_loader)):
             if len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                 tracker.step(frame, i)
                 num_frames += 1

         results = tracker.get_results()

         time_total += time.time() - start

         _log.info(f"Tracks found: {len(results)}")
         _log.info(f"Runtime for {seq}: {time.time() - start :.1f} s.")

         if tracktor['interpolate']:
             results = interpolate(results)

         if seq.no_gt:
             _log.info(f"No GT data for evaluation available.")
         else:
             mot_accums.append(get_mot_accum(results, seq))

         _log.info(f"Writing predictions to: {output_dir}")

         #seq.write_results(results, output_dir)


         if tracktor['write_images']:
             plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

    _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
               f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")

    if mot_accums:
        summary = evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)
        #summary = evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt and "04" in str(s) or "02" in str(s)], generate_overall=True)
        summary.to_pickle("output/finetuning_results/results_{}_{}_{}_{}_{}.pkl".format(tracktor['output_subdir'],
                                                                                           tracktor['tracker']['finetuning']['max_displacement'],
                                                                                            tracktor['tracker']['finetuning']['batch_size'],
                                                                                            tracktor['tracker']['finetuning']['learning_rate'],
                                                                                            tracktor['tracker']['finetuning']['iterations']))


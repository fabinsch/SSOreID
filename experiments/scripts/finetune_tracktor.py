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
import pickle
from time import sleep
import h5py

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
    _log.setLevel(tracktor['loggerLevel']) # 5 is personal debug, 10 is debug, 20 info
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
    dataset = Datasets(tracktor['dataset'])

    def calculate_area(pos):
        w = pos[2] - pos[0]
        h = pos[3] - pos[1]
        return w*h

    db_others=None
    db_others_loaded=None
    load_others=tracktor['tracker']['finetuning']['load_others']
    fill_up=tracktor['tracker']['finetuning']['fill_up']
    if load_others:# or fill_up:
        # load db for others, without area and frame
        start_time_load = time.time()
        #database = 'db_train_2'
        database = 'db_train_mot_db'
        db_others_seq = tracktor['tracker']['finetuning']['sequence_others']
        with h5py.File('./data/ML_dataset/{}.h5'.format(database), 'r') as hf:
            datasets = list(hf.keys())
            #datasets = [d for d in datasets if d != reid['dataloader']['validation_sequence']]
            db_others = {}
            #db_others = torch.tensor([])
            #db_others_id = torch.tensor([])
            db_others_area = {}
            for set in datasets:
                if db_others_seq =='ALL':
                    if set != dataset._data._data[0]._seq_name:
                        off = [t for t in sorted(db_others.keys(), reverse=True)][0] if len(db_others)>0 else 0
                        seq = hf.get(set)
                        box, data, label = seq.items()
                        for i, l in enumerate(label[1]):
                            l += off
                            #db_others = torch.cat((db_others, torch.tensor(data[1][i])))
                            #db_others_id = torch.cat((db_others_id, torch.tensor(l).float().unsqueeze(0)))

                            if l not in db_others.keys():
                                #db_others[l] = [(torch.zeros([]), torch.tensor(data[1][i]), torch.zeros([]))]
                                db_others[l] = torch.tensor(data[1][i]).unsqueeze(0)
                                #db_others_area[l] = [calculate_area(torch.tensor(box[1][i]))]
                                #print([(calculate_area(torch.tensor(box[1][i]).to(device)), torch.zeros([]).to(device))])
                            else:
                                #db_others[l].append((torch.zeros([]), torch.tensor(data[1][i]), torch.zeros([])))
                                db_others[l] = torch.cat((db_others[l], torch.tensor(data[1][i]).unsqueeze(0)))
                                #db_others_area[l].append(calculate_area(torch.tensor(box[1][i])))
                                #print([(calculate_area(torch.tensor(box[1][i]).to(device)), torch.zeros([]).to(device))])

                        _log.info('loaded {} in {} s'.format(set, time.time()-start_time_load))
                    else:
                        _log.info('do not take {}'.format(set))

                elif set == db_others_seq:
                    seq = hf.get(set)
                    box, data, label = seq.items()
                    for i, l in enumerate(tqdm(label[1])):
                        if l not in db_others.keys():
                            #b_others[l] = [(torch.zeros([]), torch.tensor(data[1][i]), torch.zeros([]))]
                            db_others[l] = torch.tensor(data[1][i]).unsqueeze(0)
                            #db_others_area[l] = [calculate_area(torch.tensor(box[1][i]))]
                            #print([(calculate_area(torch.tensor(box[1][i]).to(device)), torch.zeros([]).to(device))])
                        else:
                            #db_others[l].append((torch.zeros([]), torch.tensor(data[1][i]), torch.zeros([])))
                            db_others[l] = torch.cat((db_others[l], torch.tensor(data[1][i]).unsqueeze(0)))
                            #db_others_area[l].append(calculate_area(torch.tensor(box[1][i])))
                            #print([(calculate_area(torch.tensor(box[1][i]).to(device)), torch.zeros([]).to(device))])

                    #db_others = torch.cat((db_others,  torch.tensor(data[1])))
                    #db_others_id = torch.cat((db_others_id, torch.tensor(label[1]).float().unsqueeze(0)))
                    _log.info('loaded {} in {} s'.format(set, time.time()-start_time_load))

                    # for id in db_others.values():
                    #     for frame in id:
                    #         print(frame[0].item())
                    #         if len(frame[0].shape)>0:
                    #             print('problem')

    #print('always others class DBs built')
            print('always others class DBs loaded from {}, in total {} IDs in DB'.format(db_others_seq, len(db_others)))
            db_others_loaded = (db_others, db_others_area)
            #db_others_loaded = (db_others, db_others_id)
    # object detection
    _log.info("Initializing object detector.")
    _log.debug("Initializing object detector.")

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
    #dataset = Datasets(tracktor['dataset'])

    def load_w(weights):
        try:
            a = torch.load(weights)
            print('load succesfull')
            #split = weights.split("/")
            #save = split[0]+'/'+split[1]+'/'+split[2]+'/'+split[3]+'/'+'best_reID_Network_copy.pth'
            #torch.save(a, save)
            return a
        except:
            sleep(2)
            print('load again')
            return load_w(weights)

    if tracktor['tracker']['finetuning']['for_reid']:
        if tracktor['reid_ML']:
           #a = load_w(tracktor['reid_weights'])
           a = torch.load(tracktor['reid_weights'])

    for seq in dataset:

        obj_detect = FRCNN_FPN(num_classes=2).to(device)
        obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                                              map_location=lambda storage, loc: storage))
        obj_detect.eval()

        # tracktor
        if 'oracle' in tracktor:
            tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
        else:
            tracker = Tracker(obj_detect, reid_network, tracktor['tracker'], seq._dets+'_'+seq._seq_name, a, tracktor['reid_ML'], tracktor['LR_ML'], db=db_others_loaded)

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
        _log.info(f"{tracker.correct_reID} correct reIDs  {tracker.wrong_reID} wrong reIDs")
        _log.info(f"{tracker.correct_no_reID} correct NO reIDs , new track initialized")
        _log.info(f"{tracker.missed_reID} missed reIDs: {tracker.missed_reID_others} others, {tracker.missed_reID_score} score, {tracker.missed_reID_patience} patience")
        _log.info(f"{tracker.missed_reID_score_iou} missed reIDs scores iou, {tracker.correct_no_reID_iou} correct no reIDs scores iou")
        _log.info(f"Total number of predictions {tracker.number_made_predictions}: {tracker.correct_reID+tracker.correct_no_reID} correct, {tracker.missed_reID+tracker.wrong_reID} wrong")
        _log.info(f"How long reID tracks were inactive {tracker.inactive_count_succesfull_reID}")
        if len(tracker.inactive_count_succesfull_reID) > 0:
            _log.info(f"average is {sum(tracker.inactive_count_succesfull_reID) / len(tracker.inactive_count_succesfull_reID)}")


        _log.info(f"Total number of Trainings: {tracker.num_training}")
        _log.info(f"Number of skipped samples because of IoU restriction others: {tracker.c_skipped_for_train_iou}")
        #_log.info(f"Number of skipped samples because of IoU restriction (with just one active frame): {tracker.c_skipped_and_just_and_frame_active}")
        #_log.info(f"It was trained on: {tracker.train_on}")
        _log.info(f"It happen x times that it was killed and reid in same step: {tracker.count_killed_this_step_reid}")
        _log.info(f"Number of tracks which are just active 1 frame: {tracker.c_just_one_frame_active}")

        _log.info(f"Print statistics of training situations")
        _log.info(f"nWays {tracker.count_nways}")
        _log.info(f"kShots {tracker.count_kshots}")
        _log.info(f"TRAIN Acc values after training {tracker.acc_after_train}")
        _log.info(f"TRAIN Acc avg. {sum(tracker.acc_after_train)/len(tracker.acc_after_train)}")
        _log.info(f"VAL Acc values after training {tracker.acc_val_after_train}")
        if len(tracker.acc_val_after_train)>0:
            _log.info(f"VAL Acc avg. {sum(tracker.acc_val_after_train)/len(tracker.acc_val_after_train)}")
        _log.info(f"Currently always calculates scores for train samples - DEACTIVATE l. 1205")

        with open(seq._seq_name + '_count_nWays'+ '.pkl', 'wb') as f:
            pickle.dump(tracker.count_nways, f, pickle.HIGHEST_PROTOCOL)

        with open(seq._seq_name + '_count_nKshots'+ '.pkl', 'wb') as f:
            pickle.dump(tracker.count_kshots, f, pickle.HIGHEST_PROTOCOL)

        if len(tracker.score_others)>0:
            _log.info(f"Average score for others class: {sum(tracker.score_others) / len(tracker.score_others)}")

        if len(tracker.trained_epochs)>0:
            _log.info(f"Epochs trained effectivly: {tracker.trained_epochs}")
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

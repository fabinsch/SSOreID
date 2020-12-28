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

from tracktor.config import get_output_dir, get_tb_dir
import random
from ml.reid_model import reID_Model
from ml.maml import MAML
from ml.meta_sgd import MetaSGD
from ml.utils import load_dataset, get_ML_settings, get_plotter, save_checkpoint, sample_task

from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.datasets.factory import Datasets
from tracktor.tracker import Tracker
from torch.utils.data import DataLoader
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

ex = Experiment()
ex.add_config('experiments/cfgs/ML_reid.yaml')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ex.automain
def my_main(tracktor, reid, _config, _log, _run):
    print('both just 10 percent val and train set others own')
    print('NOT SAVING ANY MODEL')

    idf1_trainOthers = []
    idf1_NotrainOthers = []

    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(reid['seed'])
    torch.cuda.manual_seed(reid['seed'])
    np.random.seed(reid['seed'])
    torch.backends.cudnn.deterministic = True
    random.seed(reid['seed'])

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
    meta_datasets, validation_set, i_to_dataset, others_FM = load_dataset(reid, exclude=['cuhk03', 'market1501'], only='MOT17-13')
    nways_list, kshots_list, num_tasks, num_tasks_val, adaptation_steps, meta_batch_size, lr = get_ML_settings(reid)

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

    if reid['ML']['maml']:
        model = MAML(reID_network, lr=1e-3, first_order=True, allow_nograd=True)
        opt = torch.optim.Adam([
            {'params': model.module.head.parameters()},
            {'params': model.module.predictor.parameters()}
        ], lr=lr)
    if reid['ML']['learn_LR']:
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
            {'params': model.lrs, 'lr': lr_lr},  # the template + others LR
            {'params': model.module.lrs[:4], 'lr': lr_lr}  # LRs for the head
        ], lr=lr)

    ##################
    # Begin training #
    ##################
    print("[*] Training ...")
    clamping_LR = True
    if clamping_LR:
        print('clamp LR < 0')

    # how to sample, uniform from the 3 db or uniform over sequences
    if ('market1501' and 'cuhk03') in i_to_dataset.values():
        sample_db = reid['ML']['db_sample_uniform']
    else:
        sample_db = False
        print(f"WORK without market und cuhk")


    info = (nways_list, kshots_list, meta_batch_size, num_tasks, num_tasks_val,
            reid['dataloader']['validation_sequence'], reid['ML']['flip_p'])

    plotter = get_plotter(reid, info)
    plotter.init_statistics(meta_batch_size)

    for iteration in range(1, _config['reid']['solver']['iterations']+1):

        opt.zero_grad()
        plotter.reset_batch_stats()
        # 1 validation task per meta batch size train tasks
        # Compute meta-validation loss - validation sequence
        # here no backward in outer loop, just inner
        if len(validation_set) > 0:
            learner = model.clone()
            batch, nways, kshots, _, taskID = sample_task(validation_set, nways_list, kshots_list, i_to_dataset,
                                                          reid['ML']['sample_from_all'], val=True,
                                                          num_tasks=num_tasks_val)

            batch, used_labels = batch
            others_own_idx = [
                idx for l, idx in validation_set[0].labels_to_indices.items()
                if l not in used_labels]
            others_own_idx = functools.reduce(operator.iconcat, others_own_idx, [])

            evaluation_loss, evaluation_accuracy = reID_network.fast_adapt(batch=batch,
                                                                 learner=learner,
                                                                 adaptation_steps=adaptation_steps,
                                                                 shots=kshots,
                                                                 ways=nways,
                                                                 train_task=False,
                                                                 plotter=plotter,
                                                                 iteration=iteration,
                                                                 task=-1,
                                                                 taskID=taskID,
                                                                 reid=reid,
                                                                 others_own=others_own_idx,
                                                                 #others=#others_FM,
                                                                 train_others=reid['ML']['train_others'],
                                                                 set=validation_set[0])

            plotter.update_batch_val_stats(evaluation_accuracy, evaluation_loss)

        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = model.clone()  # back-propagating losses on the cloned module will populate the buffers of the original module
            batch, nways, kshots, sequence, taskID = sample_task(meta_datasets, nways_list, kshots_list, i_to_dataset,
                                                                 reid['ML']['sample_from_all'], num_tasks=num_tasks,
                                                                 sample_uniform_DB=sample_db)
            sequence, sequence_idx = sequence
            batch, used_labels = batch
            others_own_idx = [
                idx for label, idx in meta_datasets[sequence_idx].labels_to_indices.items()
                if label not in used_labels]
            others_own_idx = functools.reduce(operator.iconcat, others_own_idx, [])


            evaluation_loss, evaluation_accuracy = reID_network.fast_adapt(batch=batch,
                                                                                learner=learner,
                                                                                adaptation_steps=adaptation_steps,
                                                                                shots=kshots,
                                                                                ways=nways,
                                                                                train_task=True,
                                                                                plotter=plotter,
                                                                                iteration=iteration,
                                                                                task=task,
                                                                                taskID=taskID,
                                                                                reid=reid,
                                                                                others_own=others_own_idx,
                                                                                #others=others_FM[others_seq_ID != sequence_idx],
                                                                                seq=sequence,
                                                                                train_others=reid['ML']['train_others'],
                                                                                set=meta_datasets[sequence_idx])

            evaluation_loss.backward()  # compute gradients, populate grad buffers of maml
            plotter.update_batch_train_stats(evaluation_accuracy, evaluation_loss)
            plotter.update_statistics(model)


        # update the best value for loss
        # if len(validation_set) > 0:
        #     safe_best_loss = loss_meta_val.update_best(loss_meta_val.avg, iteration)
        # else:
        #     safe_best_loss = loss_meta_val.update_best(loss_meta_train.avg, iteration)

        if iteration % 10 == 0:
            # Print some metrics
            print(f"\nIteration {iteration}")
            plotter.print_statistics()
            if reid['solver']["plot_training_curves"] and (iteration % 100 == 0 or iteration == 1):
                plotter.plot_statistics(iteration)

        # Average the accumulated gradients
        for p in [p for p in model.parameters() if p.requires_grad]:
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_batch_size)

        opt.step()

        if clamping_LR:
            # clamping all lrs
            lrs = [p for name, p in model.named_parameters() if 'lrs' in name]
            for i, lr in enumerate(lrs):
                if (lr < 0).sum().item() > 0:
                    lr_mask = (lr > 0)
                    lr.data = lr * lr_mask


        # when working with val set
        # if iteration > 1000 and safe_best_loss == 1:
        #     if iteration % 10 == 0:
        #         model_s = '{}_reID_Network.pth'.format(iteration)
        #         if safe_best_loss:
        #             model_s = 'best_reID_Network.pth'
        #
        if iteration % 5000 == 0:
            model_s = '{}_reID_Network.pth'.format(iteration)
            model_name = osp.join(output_dir, model_s)

            state_dict_to_safe = reID_Model.remove_last(model.state_dict(), model.module.num_output)
            save_checkpoint({
                'epoch': iteration,
                'state_dict': state_dict_to_safe,
                #'best_acc': (acc_meta_train.avg, acc_meta_val.avg),
                'best_acc': (0, 0),
                #'optimizer': opt.state_dict(),
                'optimizer': None,
            }, filename=model_name)




        # evaluate tracktor to see performance when using meta-learned model
        #evaluate_tracktor = [1, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
        #evaluate_tracktor = [20000]
        #if iteration in evaluate_tracktor:
        if iteration % 5000 == 0:
            if reid['ML']['train_others']:
                trainOthers_tracktor = [True]
            else:
                trainOthers_tracktor = [True, False]
            for train_mode in trainOthers_tracktor:
                print(f'evaluate tracktor and train others: {train_mode}')
                time_total, num_frames = 0, 0
                mot_accums = []
                dataset = Datasets(tracktor['dataset'])
                a = reID_Model.remove_last(model.state_dict(), model.module.num_output)
                for seq in dataset:

                    obj_detect = FRCNN_FPN(num_classes=2).to(device)
                    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                                                          map_location=lambda storage, loc: storage))
                    obj_detect.eval()

                    # tracktor
                    tracker = Tracker(obj_detect, None, tracktor['tracker'], seq._dets + '_' + seq._seq_name, a,
                                      tracktor['reid_ML'], tracktor['LR_ML'], reid['ML']['weightening'],
                                      train_mode)

                    start = time.time()

                    _log.info(f"Tracking: {seq}")

                    data_loader = DataLoader(seq, batch_size=1, shuffle=False)
                    for i, frame in enumerate(data_loader):
                        if i >= 0 and len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                            tracker.step(frame, i)
                            num_frames += 1

                    results = tracker.get_results()

                    time_total += time.time() - start

                    _log.info(f"Tracks found: {len(results)}")
                    _log.info(f"Runtime for {seq}: {time.time() - start :.1f} s.")
                    _log.info(f"Total number of reIDs: {tracker.num_reids}")
                    _log.info(f"Total number of Trainings: {tracker.num_training}")
                    #_log.info(f"TRAIN Acc values after training {tracker.acc_after_train}")
                    # _log.info(f"TRAIN Acc avg. {sum(tracker.acc_after_train)/len(tracker.acc_after_train)}")
                    train_acc_id = [a[0] for a in tracker.acc_after_train]
                    train_acc_others = [a[1] for a in tracker.acc_after_train]
                    _log.info(f"TRAIN Acc avg. {sum(train_acc_id) / len(train_acc_id)}")
                    _log.info(f"TRAIN Acc avg. {sum(train_acc_others) / len(train_acc_others)}")
                    if len(tracker.acc_val_after_train) > 0:
                        val_acc_id = [a[0] for a in tracker.acc_val_after_train]
                        val_acc_others = [a[1] for a in tracker.acc_val_after_train]
                        _log.info(f"TRAIN Acc IDs avg. {sum(val_acc_id) / len(val_acc_id)}")
                        _log.info(f"TRAIN Acc others avg. {sum(val_acc_others) / len(val_acc_others)}")
                    #_log.info(f"VAL Acc values after training {tracker.acc_val_after_train}")
                    # if len(tracker.acc_val_after_train)>0:
                    #    _log.info(f"VAL Acc avg. {sum(tracker.acc_val_after_train)/len(tracker.acc_val_after_train)}")

                    if len(tracker.trained_epochs) > 0:
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
                del a
                if mot_accums:
                    summary = evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt],
                                                  generate_overall=True)
                    summary.to_pickle(
                        "output/finetuning_results/results_{}_{}_{}_{}_{}.pkl".format(tracktor['output_subdir'],
                                                                                      tracktor['tracker']['finetuning'][
                                                                                          'max_displacement'],
                                                                                      tracktor['tracker']['finetuning'][
                                                                                          'batch_size'],
                                                                                      tracktor['tracker']['finetuning'][
                                                                                          'learning_rate'],
                                                                                      tracktor['tracker']['finetuning'][
                                                                                          'epochs']))
                overall = summary.idf1['OVERALL'] * 100
                if train_mode:
                    plotter.plot(epoch=iteration, loss=-1, acc=overall, split_name='idf1_TrainOn')
                    idf1_trainOthers.append(round(overall,1))
                else:
                    plotter.plot(epoch=iteration, loss=-1, acc=overall, split_name='idf1_TrainOff')
                    idf1_NotrainOthers.append(round(overall, 1))

                print(f"\nIteration {iteration}")
                print(f"IDF1 scores train others in tracktor {idf1_trainOthers}")
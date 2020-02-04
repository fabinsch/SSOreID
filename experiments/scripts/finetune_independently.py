import pickle

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

from collections import defaultdict
import numpy as np
from torch.nn import functional as F
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.live_dataset import IndividualDataset
import torch
from sacred import Experiment
import sklearn.metrics

from tracktor.visualization import VisdomLinePlotter

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')
ex.add_named_config('cfg_classification', 'experiments/cfgs/cfg_classification.yaml')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_nets(obj_detect_weights):
    obj_detect = FRCNN_FPN(num_classes=2).to(device)
    obj_detect.load_state_dict(torch.load(obj_detect_weights,
                                              map_location=lambda storage, loc: storage))
    obj_detect.eval()

    bbox_predictor_weights = obj_detect.roi_heads.box_predictor.state_dict()
    bbox_head_weights = obj_detect.roi_heads.box_head.state_dict()

    box_predictor_classification = FastRCNNPredictor(1024, 2).to(device)
    box_predictor_classification.load_state_dict(bbox_predictor_weights)

    box_head_classification = TwoMLPHead(obj_detect.backbone.out_channels *
                                         obj_detect.roi_heads.box_roi_pool.output_size[0] ** 2,
                                         representation_size=1024).to(device)
    box_head_classification.load_state_dict(bbox_head_weights)
    return obj_detect, box_head_classification, box_predictor_classification


def do_finetuning(id, finetuning_config, plotter, box_head_classification, box_predictor_classification, sequence_number,
                  num_frames_train=15, num_frames_val=10, train_val_frame_gap=5, dataset=None,
                  val_data=None, train_data=None):
    validation_set = val_data
    training_set = train_data
    if not val_data or not train_data:
        dataset = dataset
        if not dataset:
            dataset = pickle.load(open("training_set/{}/feature_training_set_track_{}.pkl".format(sequence_number, id), "rb"))
            dataset.post_process()

        training_set, validation_set = dataset.val_test_split(num_frames_train=num_frames_train,
                                                              num_frames_val=num_frames_val,
                                                              train_val_frame_gap=train_val_frame_gap,
                                                              downsampling=False,
                                                              shuffle=True)
    box_predictor_classification.train()
    box_head_classification.train()
    optimizer = torch.optim.Adam(
                list(box_predictor_classification.parameters()) + list(box_head_classification.parameters()), lr=float(finetuning_config["learning_rate"]) )
    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=512)
    val_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=512)

    for i in range(int(finetuning_config["iterations"])):
        for i_sample, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = forward_pass_for_classifier_training(sample_batch['features'], sample_batch['scores'],
                                                        box_head_classification, box_predictor_classification)
            loss.backward()
            optimizer.step()
        plot_every = 1
        if np.mod(i, plot_every) == 0 and (
                finetuning_config["early_stopping_classifier"] or finetuning_config["plot_training_curves"]):
            positive_scores = torch.Tensor([]).to(device)
            negative_scores = torch.Tensor([]).to(device)
            for i_sample, sample_batch in enumerate(train_dataloader):
                positive_scores = torch.cat((positive_scores, forward_pass_for_classifier_training(
                    sample_batch['features'][sample_batch['scores'] == 1], sample_batch['scores'], box_head_classification,
                    box_predictor_classification, return_scores=True,
                    eval=True)))
                negative_scores = torch.cat((negative_scores, forward_pass_for_classifier_training(
                    sample_batch['features'][sample_batch['scores'] == 0], sample_batch['scores'], box_head_classification,
                    box_predictor_classification, return_scores=True,
                    eval=True)))

        if finetuning_config["early_stopping_classifier"] and torch.min(positive_scores) > 0.99 and torch.min(positive_scores) - torch.max(negative_scores) > 0.99:
            break

        if np.mod(i, plot_every) == 0 and finetuning_config["validate"]:
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                if finetuning_config["validate"]:
                    val_positive_scores = forward_pass_for_classifier_training(
                        val_batch['features'][val_batch['scores'] == 1], val_batch['scores'], box_head_classification, box_predictor_classification, return_scores=True,
                        eval=True)
                    val_negative_scores = forward_pass_for_classifier_training(
                        val_batch['features'][val_batch['scores'] == 0], val_batch['scores'], box_head_classification, box_predictor_classification, return_scores=True,
                        eval=True)

                if finetuning_config["validate"]:
                    val_positive_scores = val_positive_scores[:10]
                    val_negative_scores = val_negative_scores[:10]
                    for sample_idx, score in enumerate(val_positive_scores):
                        plotter.plot('score', 'val positive {}'.format(sample_idx), 'Scores Evaluation Classifier for Track {}'.format(id),
                                     i, score.cpu().numpy(), val_positive=True) #light red
                    for sample_idx, score in enumerate(val_negative_scores):
                        plotter.plot('score', 'val negative {}'.format(sample_idx), 'Scores Evaluation Classifier for Track {}'.format(id),
                                     i, score.cpu().numpy(), val_negative=True) #light blue

        box_predictor_classification.eval()
        box_head_classification.eval()

    total_samples = 0
    loss = 0
    true_labels = torch.tensor([])
    predicted_labels = torch.tensor([])
    predicted_scores = torch.tensor([])
    reid_results = []
    for idx, batch in enumerate(val_dataloader):
        new_true_scores = batch['scores'].to('cpu')
        true_labels = torch.cat([true_labels, new_true_scores])
        new_predicted_scores = forward_pass_for_classifier_training(
            batch['features'], batch['scores'], box_head_classification, box_predictor_classification,
            return_scores=True,
            eval=True)
        new_predicted_labels = torch.zeros_like(new_predicted_scores)
        new_predicted_labels[new_predicted_scores > 0.5] = 1
        new_predicted_labels[new_predicted_scores < 0.5] = 0
        predicted_labels = torch.cat([predicted_labels, new_predicted_labels.to('cpu')])
        predicted_scores = torch.cat([predicted_scores, new_predicted_scores.to('cpu')])
        total_samples += batch['features'].size()[0]


    f1_score = sklearn.metrics.f1_score(true_labels, predicted_labels)
    precision = sklearn.metrics.precision_score(true_labels, predicted_labels)
    recall = sklearn.metrics.recall_score(true_labels, predicted_labels)

    print('F1 Score for track {}: {}'.format(id, f1_score))
    return f1_score, precision, recall, reid_results


def forward_pass_for_classifier_training(features, scores, box_head_classification, box_predictor_classification, eval=False, return_scores=False):
    if eval:
        box_predictor_classification.eval()
        box_head_classification.eval()
    feat = box_head_classification(features)
    class_logits, _ = box_predictor_classification(feat)
    if return_scores:
        pred_scores = F.softmax(class_logits, -1)
        if eval:
            box_predictor_classification.train()
            box_head_classification.train()
        return pred_scores[:, 1:].squeeze(dim=1).detach()
    loss = F.cross_entropy(class_logits, scores.long())
    if eval:
        box_predictor_classification.train()
        box_head_classification.train()
    return loss

def reid_exp(finetuning_config, obj_detect_weights, sequence_number, reid_tuples):
    f1_scores = []
    pre_rec_per_frame_number = defaultdict(list)
    max_frame_number_train = 1

    for reid_tuple in reid_tuples:
        plotter = VisdomLinePlotter(env_name='finetune_independently')
        color = np.array([[np.random.randint(0, 255),  np.random.randint(0, 255),  np.random.randint(0, 255)], ])
        original_track_id = reid_tuple[0]
        new_track_id = reid_tuple[1]
        first_dataset = pickle.load(open("training_set/{}/feature_training_set_track_{}.pkl".format(sequence_number, original_track_id), "rb"))
        first_dataset.post_process()
        second_dataset = pickle.load(open("training_set/{}/feature_training_set_track_{}.pkl".format(sequence_number, new_track_id), "rb"))
        second_dataset.post_process()
        print(second_dataset.samples_per_frame)

        if len(first_dataset.samples_per_frame) <  max_frame_number_train:
            print("skipping track {}".format(original_track_id))
            continue
        for num_frame_train in [10]:#range(1, max_frame_number_train + 1, 2):
            num_frames = 40 if len(first_dataset.samples_per_frame) > 40 else len(first_dataset.samples_per_frame)
            training_set, val = first_dataset.val_test_split(num_frames_train=num_frames,
                                                           num_frames_val=0,
                                                           train_val_frame_gap=0,
                                                           downsampling=False, shuffle=True)
            _, validation_set = second_dataset.val_test_split(num_frames_train=0,
                                                              num_frames_val=1,
                                                              train_val_frame_gap=0,
                                                              downsampling=False, shuffle=True)

            obj_detect, box_head_classification, box_predictor_classification = initialize_nets(obj_detect_weights)
            f1_score, precision, recall, reid_result = do_finetuning(original_track_id, finetuning_config, plotter, box_head_classification,
                                     box_predictor_classification, sequence_number, val_data=validation_set, train_data=training_set)
            #plotter.plot('precision', f"Track {original_track_id}", 'positive examples vs. precision', num_frame_train,
            #             precision,
            #             color=color)
            #plotter.plot('recall', f"Track {original_track_id}", 'positive examples vs. recall', num_frame_train, recall,
            #             color=color)
            f1_scores.append(f1_score)
            pre_rec_per_frame_number[num_frame_train].append((precision, recall))

    """plotter = VisdomLinePlotter(env_name='finetune_independently', xlabel="number of positive examples")
    for frame_number in pre_rec_per_frame_number.keys():
        plotter.plot('avg prec score', "f1 score", 'positive examples vs. avg precision score', frame_number,
                     np.mean([m[0] for m in pre_rec_per_frame_number[frame_number]]))
        plotter.plot('avg recall score', "f1 score", 'positive examples vs. avg recall score', frame_number,
                     np.mean([m[1] for m in pre_rec_per_frame_number[frame_number]]))"""

def frame_number_train_exp(finetuning_config, obj_detect_weights, sequence_number):
    f1_scores = []
    reid_results = []
    plotter = VisdomLinePlotter(env_name='finetune_independently', xlabel="number of positive examples")
    max_frame_number_train = 60
    # skipped tracks: 79, 17, 40
    track_ids = [49, 3, 52, 57, 45, 0, 39, 23]#[19, 21, 35, 82, 79, 44, 52, 80, 47, 65, 11, 41, 42, 17, 40, 18]
    pre_rec_per_frame_number = defaultdict(list)
    for track_id in track_ids:
        color = np.array([[np.random.randint(0, 255),  np.random.randint(0, 255),  np.random.randint(0, 255)], ])
        dataset = pickle.load(open("training_set/feature_training_set_track_{}.pkl".format(track_id), "rb"))
        dataset.post_process()
        if len(dataset.samples_per_frame) < max_frame_number_train + 20 + 5:
            print("skipping track {}".format(track_id))
            continue
        for num_frames_train in range(1, max_frame_number_train + 1, 2):
            obj_detect, box_head_classification, box_predictor_classification = initialize_nets(obj_detect_weights)
            f1_score, precision, recall, reid_result = do_finetuning(track_id, finetuning_config, plotter, box_head_classification,
                                     box_predictor_classification, sequence_number, num_frames_train=num_frames_train, num_frames_val=20,
                                     train_val_frame_gap=5, dataset=dataset)
            plotter.plot('precision', f"Track {track_id}", 'positive examples vs. precision', num_frames_train, precision,
                         color=color)
            plotter.plot('recall', f"Track {track_id}", 'positive examples vs. recall', num_frames_train, recall,
                         color=color)
            pre_rec_per_frame_number[num_frames_train].append((precision, recall))
            f1_scores.append(f1_score)
            reid_results.append(reid_result)

    print("average f1 score {}".format(np.mean(f1_scores)))
    true_positives = np.sum(reid_results == 1)
    false_positives = np.sum(reid_results == 0)
    precision = true_positives / (true_positives + false_positives)
    print(f"precision: {precision}")
    plotter = VisdomLinePlotter(env_name='finetune_independently', xlabel="number of positive examples")
    for frame_number in pre_rec_per_frame_number.keys():
        plotter.plot('avg prec score', "f1 score", 'positive examples vs. avg precision score', frame_number,
                     np.mean([m[0] for m in pre_rec_per_frame_number[frame_number]]))
        plotter.plot('avg recall score', "f1 score", 'positive examples vs. avg recall score', frame_number,
                     np.mean([m[1] for m in pre_rec_per_frame_number[frame_number]]))

def test_multiple_tracks(finetuning_config, obj_detect_weights, sequence_number):
    f1_scores = []
    reid_results = []

    track_ids = [3]
    for track_id in track_ids:
        if finetuning_config['validate'] or finetuning_config['plot_training_curves']:
            plotter = VisdomLinePlotter(env_name='finetune_independently')
        else:
            plotter = None
        obj_detect, box_head_classification, box_predictor_classification = initialize_nets(obj_detect_weights)

        try:
            f1_score, precision, recall, reid_result = do_finetuning(track_id, finetuning_config, plotter, box_head_classification,
                                                  box_predictor_classification, sequence_number, num_frames_train=40, num_frames_val=20,
                                                  train_val_frame_gap=80)
            f1_scores.append(f1_score)
        except AssertionError:
            continue
    print("Track id with lowest score: {}, score: {}".format(track_ids[np.argmin(f1_scores)], np.min(f1_scores)))
    print("average f1 score {}".format(np.mean(f1_scores)))
    true_positives = np.sum(reid_results == 1)
    false_positives = np.sum(reid_results == 0)
    precision = true_positives / (true_positives + false_positives)
    print(f"precision: {precision}")



@ex.automain
def main(tracktor, _config, _log, _run):
    sequence_number = 13
    reid_tuples_02 = [(49, 90), (3, 74), (52, 81), (57, 84), (45, 79), (39, 46), (93, 104), (79, 89), (18, 56), (41, 54),
                  (1, 28), (2, 29), (33, 76)]
    reid_tuples_13 = [
        (14, 32), #gap 0,
        (40, 45), # gap 6
        (12, 36), # gap 9
        (8, 27), # gap 0
        (29, 35), # gap 1
        (90, 107), # gap 3
        (97, 108), #gap 3
    ]
    tracker_cfg = tracktor['tracker']
    finetuning_config = tracker_cfg['finetuning']
    obj_detect_weights = _config['tracktor']['obj_detect_model']

    reid_exp(finetuning_config, obj_detect_weights, sequence_number, reid_tuples_13)

# ([[1446.5616,  528.2750, 1467.5509,  582.6155]], device='cuda:0')
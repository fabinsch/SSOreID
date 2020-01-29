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
ex.add_named_config('cfg1', 'experiments/cfgs/hp_search/cfg1.yaml')

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

def get_reid_datasets(first_id, second_id, num_frames_train):
    first_dataset = pickle.load(open("training_set/feature_training_set_track_{}.pkl".format(first_id), "rb"))
    first_dataset.post_process()
    second_dataset = pickle.load(open("training_set/feature_training_set_track_{}.pkl".format(second_id), "rb"))
    second_dataset.post_process()

    training_set, _ = first_dataset.val_test_split(num_frames_train=num_frames_train,
                                                   num_frames_val=0,
                                                   train_val_frame_gap=0,
                                                   downsampling=False, shuffle=True)

    _, validation_set = second_dataset.val_test_split(num_frames_train=0,
                                                      num_frames_val=3,
                                                          train_val_frame_gap=0,
                                                          downsampling=False, shuffle=False)
    return (training_set, validation_set)

def do_finetuning(id, finetuning_config, plotter, box_head_classification, box_predictor_classification,
                  num_frames_train=15, num_frames_val=10, train_val_frame_gap=5, dataset=None,
                  val_data=None, train_data=None):
    validation_set = val_data
    training_set = train_data
    if not val_data or not train_data:
        dataset = dataset
        if not dataset:
            dataset = pickle.load(open("training_set/feature_training_set_track_{}.pkl".format(id), "rb"))
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, finetuning_config["decay_every"], gamma=finetuning_config['gamma'])
    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=512)
    val_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=512)

    for i in range(int(finetuning_config["iterations"])):
        for i_sample, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = forward_pass_for_classifier_training(sample_batch['features'], sample_batch['scores'], box_head_classification, box_predictor_classification)
            loss.backward()
            optimizer.step()
            scheduler.step()
        plot_every = 2
        if np.mod(i, plot_every) == 0 and (finetuning_config["early_stopping_classifier"] or finetuning_config["plot_training_curves"]):

            positive_scores = forward_pass_for_classifier_training(
                sample_batch['features'][sample_batch['scores'] == 1], sample_batch['scores'], box_head_classification, box_predictor_classification, return_scores=True,
                eval=True)
            negative_scores = forward_pass_for_classifier_training(
                sample_batch['features'][sample_batch['scores'] == 0], sample_batch['scores'], box_head_classification, box_predictor_classification, return_scores=True,
                eval=True)

        if np.mod(i, plot_every) == 0 and finetuning_config["plot_training_curves"]:
            positive_scores = positive_scores[:10]
            negative_scores = negative_scores[:10]
            for sample_idx, score in enumerate(positive_scores):
                plotter.plot('score', 'positive {}'.format(sample_idx), 'Scores Evaluation Classifier for Track {}'.format(id),
                             i, score.cpu().numpy(), train_positive=True) # dark red
            for sample_idx, score in enumerate(negative_scores):
                plotter.plot('score', 'negative {}'.format(sample_idx), 'Scores Evaluation Classifier for Track {}'.format(id),
                                  i, score.cpu().numpy())

        if finetuning_config["early_stopping_classifier"] and torch.min(positive_scores) - torch.max(negative_scores) > 1.5:
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
                break


        box_predictor_classification.eval()
        box_head_classification.eval()

    total_samples = 0
    loss = 0
    true_labels = torch.tensor([])
    predicted_labels = torch.tensor([])

    for idx, batch in enumerate(val_dataloader):
        new_true_scores = batch['scores'].to('cpu')
        true_labels = torch.cat([true_labels, new_true_scores])
        predicted_scores = forward_pass_for_classifier_training(
                batch['features'], batch['scores'], box_head_classification, box_predictor_classification, return_scores=True,
                eval=True)
        new_predicted_labels = predicted_scores
        new_predicted_labels[predicted_scores > 0.5] = 1
        new_predicted_labels[predicted_scores < 0.5] = 0
        predicted_labels = torch.cat([predicted_labels, new_predicted_labels.to('cpu')])
        loss += forward_pass_for_classifier_training(batch['features'], batch['scores'], box_head_classification, box_predictor_classification)
        total_samples += batch['features'].size()[0]

    print('Loss for track {}: {}'.format(id, loss / total_samples))
    f1_score = sklearn.metrics.f1_score(true_labels, predicted_labels)
    print('F1 Score for track {}: {}'.format(id, f1_score))
    return f1_score

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

def reid_exp(finetuning_config, obj_detect_weights):
    track_ids = [21]
    f1_per_frame_number = defaultdict(list)
    max_frame_number_train = 40
    f1_scores = []

    for num_frames_train in range(1, max_frame_number_train + 1):
        training_set, validation_set = get_reid_datasets(44, 76, num_frames_train)

        obj_detect, box_head_classification, box_predictor_classification = initialize_nets(obj_detect_weights)
        f1_score = do_finetuning(44, finetuning_config, None, box_head_classification,
                                 box_predictor_classification, val_data=validation_set, train_data=training_set)
        f1_per_frame_number[num_frames_train].append(f1_score)
        f1_scores.append(f1_score)

    print("average f1 score {}".format(np.mean(f1_scores)))
    plotter = VisdomLinePlotter(env_name='finetune_independently', xlabel="number of positive examples")
    for frame_number in f1_per_frame_number.keys():
        plotter.plot('avg f1 score', "f1 score", 'positive examples vs. avg f1 score', frame_number,
                     np.mean(f1_per_frame_number[frame_number]))

def frame_number_train_exp(finetuning_config, obj_detect_weights):
    f1_scores = []
    plotter = VisdomLinePlotter(env_name='finetune_independently', xlabel="number of positive examples")
    max_frame_number_train = 40
    # intersting reid 21 --> 65
    idsw_ids = [86, 14, 34, 13, 74]
    # skipped tracks: 79, 17, 40
    track_ids = [19, 21, 35, 82, 79, 44, 52, 80, 47, 65, 11, 41, 42, 17, 40, 18]
    f1_per_frame_number = defaultdict(list)
    for track_id in track_ids:
        color = np.array([[np.random.randint(0, 255),  np.random.randint(0, 255),  np.random.randint(0, 255)], ])
        dataset = pickle.load(open("training_set/feature_training_set_track_{}.pkl".format(track_id), "rb"))
        dataset.post_process()
        if len(dataset.samples_per_frame) < max_frame_number_train + 20 + 5:
            print("skipping track {}".format(track_id))
            continue
        for num_frames_train in range(1, max_frame_number_train + 1):
            obj_detect, box_head_classification, box_predictor_classification = initialize_nets(obj_detect_weights)
            f1_score = do_finetuning(track_id, finetuning_config, plotter, box_head_classification,
                                     box_predictor_classification, num_frames_train=num_frames_train, num_frames_val=20,
                                     train_val_frame_gap=5, dataset=dataset)
            plotter.plot('f1 score', f"Track {track_id}", 'positive examples vs. f1 score', num_frames_train, f1_score,
                         color=color)
            f1_per_frame_number[num_frames_train].append(f1_score)
            f1_scores.append(f1_score)
    print("average f1 score {}".format(np.mean(f1_scores)))
    plotter = VisdomLinePlotter(env_name='finetune_independently', xlabel="number of positive examples")
    for frame_number in f1_per_frame_number.keys():
        plotter.plot('avg f1 score', "f1 score", 'positive examples vs. avg f1 score', frame_number,
                     np.mean(f1_per_frame_number[frame_number]))

def test_multiple_tracks(finetuning_config, obj_detect_weights):
    f1_scores = []

    track_ids = [6]
    for track_id in track_ids:
        if finetuning_config['validate'] or finetuning_config['plot_training_curves']:
            plotter = VisdomLinePlotter(env_name='finetune_independently')
        else:
            plotter = None
        obj_detect, box_head_classification, box_predictor_classification = initialize_nets(obj_detect_weights)

        try:
            f1_score = do_finetuning(track_id, finetuning_config, plotter, box_head_classification,
                                     box_predictor_classification, num_frames_train=15, num_frames_val=10,
                                     train_val_frame_gap=5)
            f1_scores.append(f1_score)
        except AssertionError:
            continue
    print("Track id with lowest score: {}, score: {}".format(track_ids[np.argmin(f1_scores)], np.min(f1_scores)))
    print("average f1 score {}".format(np.mean(f1_scores)))


@ex.automain
def main(tracktor, _config, _log, _run):

    tracker_cfg = tracktor['tracker']
    finetuning_config = tracker_cfg['finetuning']
    obj_detect_weights = _config['tracktor']['obj_detect_model']

    frame_number_train_exp(finetuning_config, obj_detect_weights)

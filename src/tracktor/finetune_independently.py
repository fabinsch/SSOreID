import pickle

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

from torch.nn import functional as F
from tracktor.frcnn_fpn import FRCNN_FPN
import torch
import sacred
from sacred import Experiment

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

def do_finetuning(id, finetuning_config, plotter, box_head_classification, box_predictor_classification):
    training_set = pickle.load(open("training_set/feature_training_set_track_{}.pkl".format(id), "rb"))

    box_predictor_classification.train()
    box_head_classification.train()
    optimizer = torch.optim.Adam(
                list(box_predictor_classification.parameters()) + list(box_head_classification.parameters()), lr=float(finetuning_config["learning_rate"]) )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=finetuning_config['gamma'])

    for i in range(int(finetuning_config["iterations"])):
        print(i)
        optimizer.zero_grad()
        loss = forward_pass_for_classifier_training(training_set['features'], training_set['scores'], box_head_classification, box_predictor_classification)

        if finetuning_config["early_stopping_classifier"] or finetuning_config["validate"]:
            scores = forward_pass_for_classifier_training(training_set['features'], training_set['scores'], box_head_classification, box_predictor_classification,
                                                               return_scores=True, eval=True)

        if finetuning_config["validate"]:
            plotter.plot('loss', 'positive', 'Class Loss Evaluation Track {}'.format(id), i,
                              scores[0].cpu().numpy(), is_target=True)
            for sample in range(16, 32):
                plotter.plot('loss', 'negative {}'.format(sample),
                                  'Class Loss Evaluation Track {}'.format(id), i, scores[sample].cpu().numpy())


        if finetuning_config["early_stopping_classifier"] and scores[0] - torch.max(scores[16:]) > 1.5:
            print('Stopping because difference between positive score and maximum negative score is {}'.format(
                scores[0] - torch.max(scores[16:])))
            break

        loss.backward()
        optimizer.step()
        scheduler.step()

    box_predictor_classification.eval()
    box_head_classification.eval()

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


@ex.automain
def main(tracktor, _config, _log, _run):

    tracker_cfg = tracktor['tracker']
    finetuning_config = tracker_cfg['finetuning']
    obj_detect_weights = _config['tracktor']['obj_detect_model']
    plotter = VisdomLinePlotter(env_name='finetune_independently')

    obj_detect, box_head_classification, box_predictor_classification = initialize_nets(obj_detect_weights)
    track_id = 7
    do_finetuning(track_id, finetuning_config, plotter, box_head_classification, box_predictor_classification)

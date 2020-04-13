import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
from visdom import Visdom


def plot_compare_bounding_boxes(box_finetune, box_no_finetune, image):
    image = image.squeeze()
    image = image.permute(1, 2, 0)

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    colors = ['salmon', 'cyan', 'white']

    for i, box in enumerate([box_finetune, box_no_finetune]):
        box_np = box.numpy()
        ax.add_patch(
            plt.Rectangle((box_np[0, 0], box_np[0, 1]),
                          box_np[0, 2] - box_np[0, 0],
                          box_np[0, 3] - box_np[0, 1], fill=False,
                          linewidth=0.9, color=colors[i])
        )

    for gt_bbox in parse_ground_truth(1):
        ax.add_patch(
            plt.Rectangle((gt_bbox[0], gt_bbox[1]),
                          gt_bbox[2],
                          gt_bbox[3], fill=False,
                          linewidth=0.9, color=colors[2])
        )
    plt.axis('off')
    plt.show()
    return


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, id, env, n_samples_train, n_samples_val):
        self.viz = Visdom(port=8097, env=env)
        self.env = env
        self.id = id
        self.n_samples_train = n_samples_train
        self.n_samples_val = n_samples_val
        self.loss_window = self.viz.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='epoch',
                                     ylabel='Loss',
                                     env=self.env,
                                     title='Loss inactive {}'.format(id),
                                     legend=['train #{}'.format(self.n_samples_train)]))
        self.accuracy_window = self.viz.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='epoch',
                                     ylabel='accuracy in %',
                                     env=self.env,
                                     title='Accuracy inactive {}'.format(id),
                                     legend=['train #{}'.format(self.n_samples_train)]))

    def plot_(self, epoch, loss, acc, split_name):
        if split_name=='train':
            name = split_name+' #{}'.format(self.n_samples_train)
        elif split_name =='val':
            name = split_name+' #{}'.format(int(self.n_samples_val*0.2))
        else:
            print('error, splitname incorrect')

        self.viz.line(
            X=torch.ones((1, 1)).cpu() * epoch,
            Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
            env=self.env,
            win=self.loss_window,
            name=name,
            update='append')
        self.viz.line(
            X=torch.ones((1, 1)).cpu() * epoch,
            Y=torch.Tensor([acc]).unsqueeze(0).cpu(),
            env=self.env,
            win=self.accuracy_window,
            name=name,
            update='append')



def plot_bounding_boxes(im_info, gt_pos, image, proposals, iteration, id, validate=False):
    num_proposals = len(proposals)
    h, w = im_info
    image = image.squeeze()
    image = image.permute(1, 2, 0)
    fig, ax = plt.subplots()
    plt.imshow(image, cmap='gist_gray_r')
    gt_pos_np = gt_pos.cpu().numpy()
    for i in range(num_proposals):
        ax.add_patch(
            plt.Rectangle((proposals[i, 0], proposals[i, 1]),
                          proposals[i, 2] - proposals[i, 0],
                          proposals[i, 3] - proposals[i, 1], fill=False,
                          linewidth=0.2, color='yellow' if proposals[i, 4] == 1 else "red")
        )

    ax.add_patch(
        plt.Rectangle((gt_pos_np[0, 0], gt_pos_np[0, 1]),
                      gt_pos_np[0, 2] - gt_pos_np[0, 0],
                      gt_pos_np[0, 3] - gt_pos_np[0, 1], fill=False,
                      linewidth=0.2, color='salmon')
    )
    """
    for gt_bbox in parse_ground_truth(iteration)[:1]:
        ax.add_patch(
            plt.Rectangle((gt_bbox[0], gt_bbox[1]),
                          gt_bbox[2] - gt_bbox[0],
                          gt_bbox[3] - gt_bbox[1], fill=False,
                          linewidth=0.9, color='white')
        )"""

    plt.axis('off')
    if not validate:
        plt.savefig('./training_set/training_set_{}_{}.png'.format(id, iteration),
                    dpi=800, bbox_inches='tight')
    else:
        plt.savefig('./training_set/training_progress_{}_{}.png'.format(id, iteration), dpi=800, bbox_inches='tight')

def parse_ground_truth(frame, dets):
    bounding_boxes_xywh = []
    for i, det in dets.iterrows():
        if int(det[0]) == frame:
            bounding_boxes_xywh.append(np.array([det[2], det[3], det[4], det[5]]))

    bounding_boxes_x1x2_torch = torch.tensor(transform_to_x1y1x2y2(np.array(bounding_boxes_xywh)))
    return bounding_boxes_x1x2_torch


def transform_to_x1y1x2y2(training_boxes_xywh):
    training_boxes = training_boxes_xywh
    training_boxes[:, 2] = training_boxes_xywh[:, 0] + training_boxes_xywh[:, 2]
    training_boxes[:, 3] = training_boxes_xywh[:, 1] + training_boxes_xywh[:, 3]
    return training_boxes


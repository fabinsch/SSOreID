import numpy as np
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
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def plot_bounding_boxes(im_info, gt_pos, image, proposals, ax=None):
    num_proposals = len(proposals)
    h, w = im_info
    image = image.squeeze()
    image = image.permute(1, 2, 0)
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gist_gray_r')
    gt_pos_np = gt_pos.numpy()
    for i in range(num_proposals):
        ax.add_patch(
            plt.Rectangle((proposals[i, 0], proposals[i, 1]),
                          proposals[i, 2] - proposals[i, 0],
                          proposals[i, 3] - proposals[i, 1], fill=False,
                          linewidth=0.9, color='yellow')
        )
    ax.add_patch(
        plt.Rectangle((gt_pos_np[0, 0], gt_pos_np[0, 1]),
                      gt_pos_np[0, 2] - gt_pos_np[0, 0],
                      gt_pos_np[0, 3] - gt_pos_np[0, 1], fill=False,
                      linewidth=1.5, color='salmon')
    )

    plt.axis('off')
    #plt.show()

def parse_ground_truth(frame, file_path='/home/carolin/ADLCV/tracking_wo_bnw/data/MOT17/train/MOT17-09-FRCNN/gt/gt.txt'):
    dets = pd.read_csv(file_path, header=None, sep=',')
    bounding_boxes = []
    for i, det in dets.iterrows():
        if int(det[0]) == frame:
            print(bounding_boxes)
            bounding_boxes.append(np.array([det[2], det[3], det[4], det[5]]))
    return bounding_boxes

if __name__ == '__main__':
    parse_ground_truth(1)



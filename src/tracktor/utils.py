#########################################
# Still ugly file with helper functions #
#########################################

import os
from collections import defaultdict
from os import path as osp

import numpy as np
import torch
from cycler import cycler as cy

import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import motmetrics as mm
import copy

import h5py
# import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data


from tracktor.live_dataset import IndividualDataset
from tracktor.training_set_generation import replicate_and_randomize_boxes
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops.boxes import clip_boxes_to_image
from tracktor.track import Track
matplotlib.use('Agg')
import logging

logger = logging.getLogger('main.utils')

# https://matplotlib.org/cycler/
# get all colors with
# colors = []
#	for name,_ in matplotlib.colors.cnames.items():
#		colors.append(name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1],
                                                                        query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2],
                                                                        query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def plot_sequence(tracks, db, output_dir):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """

    print("[*] Plotting whole sequence to {}".format(output_dir))

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        im_path = v['img_path']
        im_name = osp.basename(im_path)
        im_output = osp.join(output_dir, im_name)
        im = cv2.imread(im_path)
        im = im[:, :, (2, 1, 0)]

        sizes = np.shape(im)
        height = float(sizes[0])
        width = float(sizes[1])

        fig = plt.figure()
        fig.set_size_inches(width / 100, height / 100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im)

        for j, t in tracks.items():
            if i in t.keys():
                t_i = t[i]
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=1.0, **styles[j]
                    ))

                ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                            color=styles[j]['ec'], weight='bold', fontsize=40, ha='center', va='center')

        plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        plt.savefig(im_output, dpi=100)
        plt.close()


def plot_tracks(blobs, tracks, gt_tracks=None, output_dir=None, name=None):
    # output_dir = get_output_dir("anchor_gt_demo")
    im_paths = blobs['im_paths']
    if not name:
        im0_name = osp.basename(im_paths[0])
    else:
        im0_name = str(name) + ".jpg"
    im0 = cv2.imread(im_paths[0])
    im1 = cv2.imread(im_paths[1])
    im0 = im0[:, :, (2, 1, 0)]
    im1 = im1[:, :, (2, 1, 0)]

    im_scales = blobs['im_info'][0, 2]

    tracks = tracks.data.cpu().numpy() / im_scales
    num_tracks = tracks.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(im0, aspect='equal')
    ax[1].imshow(im1, aspect='equal')

    # infinte color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    ax[0].set_title(('{} tracks').format(num_tracks), fontsize=14)

    for i, t in enumerate(tracks):
        t0 = t[0]
        t1 = t[1]
        ax[0].add_patch(
            plt.Rectangle(
                (t0[0], t0[1]),
                t0[2] - t0[0],
                t0[3] - t0[1], fill=False,
                linewidth=1.0, **styles[i]
            ))

        ax[1].add_patch(
            plt.Rectangle(
                (t1[0], t1[1]),
                t1[2] - t1[0],
                t1[3] - t1[1], fill=False,
                linewidth=1.0, **styles[i]
            ))

    if gt_tracks:
        for gt in gt_tracks:
            for i in range(2):
                ax[i].add_patch(
                    plt.Rectangle(
                        (gt[i][0], gt[i][1]),
                        gt[i][2] - gt[i][0],
                        gt[i][3] - gt[i][1], fill=False,
                        edgecolor='blue', linewidth=1.0
                    ))

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    image = None
    if output_dir:
        im_output = osp.join(output_dir, im0_name)
        plt.savefig(im_output)
    else:
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def interpolate(tracks):
    interpolated = {}
    for i, track in tracks.items():
        interpolated[i] = {}
        frames = []
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        for f, bb in track.items():
            frames.append(f)
            x0.append(bb[0])
            y0.append(bb[1])
            x1.append(bb[2])
            y1.append(bb[3])

        if len(frames) > 1:
            x0_inter = interp1d(frames, x0)
            y0_inter = interp1d(frames, y0)
            x1_inter = interp1d(frames, x1)
            y1_inter = interp1d(frames, y1)

            for f in range(min(frames), max(frames) + 1):
                bb = np.array([x0_inter(f), y0_inter(f), x1_inter(f), y1_inter(f)])
                interpolated[i][f] = bb
        else:
            interpolated[i][frames[0]] = np.array([x0[0], y0[0], x1[0], y1[0]])

    return interpolated


def bbox_transform_inv(boxes, deltas):
    # Input should be both tensor or both Variable and on the same device
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.cat(
        [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,
                                  pred_ctr_y - 0.5 * pred_h,
                                  pred_ctr_x + 0.5 * pred_w,
                                  pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)
    return pred_boxes



def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    boxes must be tensor or Variable, im_shape can be anything but Variable
    """
    if not hasattr(boxes, 'data'):
        boxes_ = boxes.numpy()

    boxes = boxes.view(boxes.size(0), -1, 4)
    boxes = torch.stack([
        boxes[:, :, 0].clamp(0, im_shape[1] - 1),
        boxes[:, :, 1].clamp(0, im_shape[0] - 1),
        boxes[:, :, 2].clamp(0, im_shape[1] - 1),
        boxes[:, :, 3].clamp(0, im_shape[0] - 1)
    ], 2).view(boxes.size(0), -1)

    return boxes


def get_center(pos):
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


def get_width(pos):
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    return torch.Tensor([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]]).cuda()


def warp_pos(pos, warp_matrix):
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    warp = torch.cat((p1_n, p2_n), 1).view(1, -1)
    if torch.cuda.is_available():
        warp = warp.cuda()
    return warp


def get_mot_accum(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                   axis=1)
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall, )

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names, )
    print(str_summary)
    return summary


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
   from https://github.com/Bjarten/early-stopping-pytorch """

    def __init__(self, patience=7, verbose=False, delta=0, checkpoints={}):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoints = checkpoints
        self.epoch = 0

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.epoch = epoch

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        for i, m in enumerate(model):
            self.checkpoints[i] = copy.deepcopy(m.state_dict())
        self.val_loss_min = val_loss


class EarlyStopping2:
    def __init__(self, verbose=False, checkpoints={}, ep_safe=1):
        """
        Args:
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.checkpoints = checkpoints
        self.epoch = 0
        self.checkpoints_250 = dict()
        self.ep_safe = ep_safe

    def __call__(self, val_loss, model, epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, model)


        elif score <= self.best_score:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, model)

        if epoch == self.ep_safe:
            self.save_checkpoint(val_loss, model, safe=True)

    def save_checkpoint(self, val_loss, model, safe=False):
        if safe:
            for i, m in enumerate(model):
                self.checkpoints_250[i] = copy.deepcopy(m.state_dict())
        else:
            '''Saves model when validation loss decrease.'''
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            for i, m in enumerate(model):
                self.checkpoints[i] = copy.deepcopy(m.state_dict())
            self.val_loss_min = val_loss


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di[
                                                                                                                 'file_path'] ==
                                                                                                             removal_keys[
                                                                                                                 0] else di
                for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]

# switch from GT searching when going inactive to GT while adding
# def tracks_to_inactive_oracle(self, tracks):
#     self.tracks = [t for t in self.tracks if t not in tracks]
#     for t in tracks:
#         # t = torch.arange(self.im_index-(t.training_set.features.shape[0]))
#
#         gt_boxes = torch.cat(list(self.gt.values()), 0).to(device)
#         inactivetrack_iou_GT = bbox_overlaps(t.pos, gt_boxes).cpu().numpy()
#         ind = np.where(inactivetrack_iou_GT == np.max(inactivetrack_iou_GT))[1]
#         if len(ind) > 0:
#             ind = ind[0]
#             overlap = inactivetrack_iou_GT[0, ind]
#             if overlap >= 0.5:
#                 gt_id = list(self.gt.keys())[ind]
#                 t.gt_id = gt_id
#                 self.inactive_tracks_gt_id.append(gt_id)
#
#                 if gt_id not in self.db_gt_inactive.keys():
#                     self.db_gt_inactive[gt_id] = [t.id]
#                 else:
#                     self.db_gt_inactive[gt_id].append(t.id)
#             else:
#                 print('no GT id found for track {}, overlap {}'.format(t.id, overlap))
#
#         if t.frames_since_active > 1:
#             t.pos = t.last_pos[-1]
#             self.inactive_number_changes += 1
#             self.killed_this_step.append(t.id)
#             if t.skipped_for_train == 1:
#                 self.c_skipped_and_just_and_frame_active += 1
#         else:
#             self.c_just_one_frame_active += 1
#             # remove tracks with just 1 active frame
#             # tracks.remove(t)
#             t.pos = t.last_pos[-1]
#             self.inactive_number_changes += 1
#             self.killed_this_step.append(t.id)
#             if t.skipped_for_train == 1:
#                 self.c_skipped_and_just_and_frame_active += 1
#     self.inactive_tracks += tracks
def tracks_to_inactive_oracle(self, tracks):
    self.tracks = [t for t in self.tracks if t not in tracks]
    for t in tracks:
        if t.gt_id == None:
            logger.debug('try to get ID for track when it gets inactive')
            gt_boxes = torch.cat(list(self.gt.values()), 0).to(device)
            inactivetrack_iou_GT = bbox_overlaps(t.pos, gt_boxes).cpu().numpy()
            ind = np.where(inactivetrack_iou_GT == np.max(inactivetrack_iou_GT))[1]
            if len(ind) > 0:
                ind = ind[0]
                overlap = inactivetrack_iou_GT[0, ind]
                if overlap >= 0.2:
                    gt_id = list(self.gt.keys())[ind]
                    t.gt_id = gt_id

        if t.gt_id not in self.db_gt_inactive.keys():
            self.db_gt_inactive[t.gt_id] = [t.id]
        else:
            self.db_gt_inactive[t.gt_id].append(t.id)

        if hasattr(t, 'box_predictor_classification_debug'):
            logger.log(5, 'Statistics for track {}|{}'.format(t.id, t.correct_prediction))
            if len(t.following_scores)>0:
                print(t.following_scores)
                print(t.follwing_corr)
                scores_sum = np.zeros(t.following_scores[0].shape)
                for s in t.following_scores:
                    scores_sum = np.concatenate((scores_sum, s))
                logger.log(5, '{} mean scores {}'.format(t.correct_prediction, np.mean(scores_sum[1:], axis=0)))

            # reset
            del t.box_predictor_classification_debug
            t.following_scores = None


        if t.frames_since_active > 1:
            t.pos = t.last_pos[-1]
            self.inactive_number_changes += 1
            self.killed_this_step.append(t.id)
            if t.skipped_for_train == 1:
                self.c_skipped_and_just_and_frame_active += 1
        else:
            self.c_just_one_frame_active += 1
            # remove tracks with just 1 active frame
            # tracks.remove(t)
            t.pos = t.last_pos[-1]
            self.inactive_number_changes += 1
            self.killed_this_step.append(t.id)
            if t.skipped_for_train == 1:
                self.c_skipped_and_just_and_frame_active += 1
    self.inactive_tracks += tracks

def reid_by_finetuned_model_oracle(self, new_det_pos, new_det_scores, frame, blob):
    """Do reid with one model predicting the score for each inactive track"""
    image = blob['img'][0]
    gt = blob['gt']
    det_gt_id = []  # list which detections corresponds to which gt id
    inactive_tracks_gt_id = []
    current_inactive_tracks_id = [t.id for t in self.inactive_tracks]
    #inactive_tracks_gt_id = self.inactive_tracks_gt_id
    inactive_tracks_gt_id = [t.gt_id for t in self.inactive_tracks]
    samples_per_track = [t.training_set.num_frames_keep for t in self.inactive_tracks]
    assert current_inactive_tracks_id == self.inactive_tracks_id
    track_missed = None

    # active_tracks = self.get_pos()
    if len(new_det_pos.size()) > 1 and len(self.inactive_tracks) > 0:
        remove_inactive = []
        det_index_to_candidate = defaultdict(list)
        inactive_to_det = defaultdict(list)
        assigned = []
        inactive_tracks = self.get_pos(active=False)
        track_missed = [-1] * new_det_pos.shape[0]

        for det in new_det_pos:
            self.number_made_predictions += 1
            gt_boxes = torch.cat(list(gt.values()), 0).to(device)
            det_iou_GT = bbox_overlaps(det.unsqueeze(0), gt_boxes).cpu().numpy()
            ind = np.where(det_iou_GT == np.max(det_iou_GT))[1]
            if len(ind) > 0:
                ind = ind[0]
                overlap = det_iou_GT[0, ind]
                if overlap >= 0.2:
                    det_gt_id.append(list(gt.keys())[ind])
                else:
                    logger.debug('no GT found for new detection, overlap {}'.format(overlap))
                    det_gt_id.append(-1)

        boxes, scores = self.obj_detect.predict_boxes(new_det_pos,
                                                      box_predictor_classification=self.box_predictor_classification,
                                                      box_head_classification=self.box_head_classification,
                                                      pred_multiclass=True)

        zero_scores = [s.item() for s in scores[:, 0]] if self.finetuning_config['others_class'] else []
        for zero_score in zero_scores:
            self.score_others.append(zero_score)

        # if frame==420:
        if frame >= 0:
            logger.debug('\n{}: scores reID: {}'.format(self.im_index, scores))
            logger.debug('IDs: {} with respectively {} samples'.format(current_inactive_tracks_id, samples_per_track))
            logger.debug('GT : {} ID'.format(inactive_tracks_gt_id))

        # check if scores has very high value, don't use IoU restriction in that case
        # no_mask = torch.ge(scores, 0.95)
        # calculate IoU distances
        iou = bbox_overlaps(new_det_pos, inactive_tracks)
        if self.finetuning_config['others_class']:
            # iou has just values for the inactive tracks -> extend for others class
            iou = torch.cat((torch.ones(iou.shape[0], 1).to(device), iou), dim=1)
        if type(self.others_db) is tuple:
            k = len(self.others_db[0])
        else:
            k = len(self.others_db)
        if self.finetuning_config['fill_up'] and len(inactive_tracks)<self.fill_up_to and k>=self.fill_up_to: ## if filled up to always 10 inactive plus others
                fill = self.fill_up_to-len(inactive_tracks)
                iou = torch.cat((torch.ones(iou.shape[0], fill).to(device), iou), dim=1)

        if len(self.flexible) > 0:
            fill = self.flexible[0]
            iou = torch.cat((torch.ones(iou.shape[0], fill).to(device), iou), dim=1)

        iou_mask = torch.ge(iou, self.reid_iou_threshold)
        # scores = scores * iou_mask + scores * no_mask

        if self.box_predictor_classification.cls_score.out_features == 1:
            scores = scores * iou_mask.squeeze()
            for i, s in enumerate(scores.cpu().numpy()):
                #if s > 0.5:
                if s > self.finetuning_config['reid_score_threshold']:
                    inactive_track = self.inactive_tracks[0]
                    det_index_to_candidate[i].append((inactive_track, s))
                    inactive_to_det[0].append(i)

        else:
            # if self.im_index==138:
            #     iou_mask = torch.cat((iou_mask, torch.tensor([False]).unsqueeze(0).to(device)), dim=1)
            #     iou_mask = torch.cat((iou_mask, torch.tensor([False]).unsqueeze(0).to(device)), dim=1)
            #     #iou_mask = iou_mask
            old_scores = scores.cpu().numpy().copy()
            old_max = old_scores.max(axis=1)

            scores = scores * iou_mask
            scores = scores.cpu().numpy()
            max = scores.max(axis=1)
            max_idx = scores.argmax(axis=1)
            scores[:, max_idx] = 0
            max2 = scores.max(axis=1)
            # max_idx2 = scores.argmax(axis=1)
            dist = max - max2

            for i, d in enumerate(dist):
                if (max[i] > self.finetuning_config['reid_score_threshold']):
                    #if False: # TODO
                    if self.finetuning_config['others_class'] or (len(inactive_tracks) == 1 and not self.finetuning_config['fill_up'] and not self.finetuning_config['flexible']):
                        # idx = 0 means unknown background people, idx=1,2,.. is inactive
                        if max_idx[i] == 0:
                            logger.debug('no reid because class 0 has score {}'.format(max[i]))
                            if det_gt_id[i] in inactive_tracks_gt_id:  # check if the gt id has ever been seen before
                                f = 0
                                self.missed_reID += 1
                                self.missed_reID_others += 1
                                for j, t in enumerate(self.inactive_tracks):  # if it's still in the inactive tracks
                                    if t.gt_id == det_gt_id[i]:
                                        # self.exclude_from_others.append(t.id)
                                        logger.debug('!!!!! missed reID of person {}, detection {}'.format(t.id, i))
                                        #self.missed_reID += 1
                                        #self.missed_reID_others += 1
                                        #self.det_new_track_exclude.append(i)
                                        f += 1

                                        # track missed, j is position correct prediction
                                        if track_missed[i] == -1:
                                            track_missed[i] = [j]
                                        else:
                                            track_missed[i].append(j)

                                if f == 0:  # means this track is not in inactive anymore but was before
                                    for id in self.db_gt_inactive[det_gt_id[i]]:
                                        #self.exclude_from_others.append(id)
                                        #print("exlude {}".format(id))
                                        logger.debug("track {} not in inactive frames anymore".format(id))
                                        self.missed_reID_patience += 1
                            else:
                                logger.debug("correctly initialized new track")
                                self.correct_no_reID += 1

                        else:
                            if self.finetuning_config['fill_up']:
                                if (max_idx[i]-1) >= len(inactive_tracks):
                                    logger.debug('fill up person with highest score')
                                else:
                                    inactive_track = self.inactive_tracks[max_idx[i] - 1]
                                    det_index_to_candidate[i].append((inactive_track, max[i]))
                                    inactive_to_det[max_idx[i] - 1].append(i)
                            else:
                                inactive_track = self.inactive_tracks[max_idx[i] - 1]
                                det_index_to_candidate[i].append((inactive_track, max[i]))
                                inactive_to_det[max_idx[i] - 1].append(i)

                    else:
                        if self.finetuning_config['fill_up'] or self.finetuning_config['flexible']: # test flexible fill up
                            if max_idx[i] >= len(inactive_tracks):
                                logger.debug('fill up person with highest score')
                                if det_gt_id[
                                    i] in inactive_tracks_gt_id:  # check if the gt id has ever been seen before
                                    f = 0
                                    self.missed_reID += 1
                                    self.missed_reID_others += 1
                                    for j, t in enumerate(self.inactive_tracks):  # if it's still in the inactive tracks
                                        if t.gt_id == det_gt_id[i]:
                                            #self.exclude_from_others.append(t.id)
                                            #logger.debug("exlude {}".format(t.id))
                                            logger.debug('!!!!! missed reID of person {}, detection {}'.format(t.id, i))
                                            # self.missed_reID += 1
                                            # self.missed_reID_others += 1
                                            # self.det_new_track_exclude.append(i)
                                            f += 1

                                            # track missed, j is position correct prediction
                                            if track_missed[i] == -1:
                                                track_missed[i] = [j]
                                            else:
                                                track_missed[i].append(j)

                                    if f == 0:  # means this track is not in inactive anymore but was before
                                        for id in self.db_gt_inactive[det_gt_id[i]]:
                                            #self.exclude_from_others.append(id)
                                            #logger.debug("exlude {}".format(id))
                                            logger.debug("track {} not in inactive frames anymore".format(id))
                                            self.missed_reID_patience += 1
                                else:
                                    logger.debug("correctly initialized new track")
                                    self.correct_no_reID += 1
                            else:
                                inactive_track = self.inactive_tracks[max_idx[i]]
                                det_index_to_candidate[i].append((inactive_track, max[i]))
                                inactive_to_det[max_idx[i]].append(i)
                        else:
                            inactive_track = self.inactive_tracks[max_idx[i]]
                            det_index_to_candidate[i].append((inactive_track, max[i]))
                            inactive_to_det[max_idx[i]].append(i)

                # elif max[i] > 0.0:
                else:
                    logger.debug('no reid with score {}, old max {}'.format(max[i], old_max[i]))
                    if det_gt_id[i] in inactive_tracks_gt_id:
                        f = 0
                        self.missed_reID += 1
                        self.missed_reID_score += 1
                        if max[i]!=old_max[i]:
                            self.missed_reID_score_iou += 1
                        for j, t in enumerate(self.inactive_tracks):
                            if t.gt_id == det_gt_id[i]:
                                #self.exclude_from_others.append(t.id)
                                #logger.debug("exlude {}".format(t.id))
                                logger.debug('!!!!! missed reID of person {}'.format(t.id))
                                #self.missed_reID += 1
                                #self.det_new_track_exclude.append(i)
                                #self.missed_reID_score += 1

                                # track missed, j is position correct prediction
                                if track_missed[i] == -1:
                                    track_missed[i] = [j]
                                else:
                                    track_missed[i].append(j)

                                # attach classifier
                                # t.add_classifier(box_predictor_classification=self.box_predictor_classification,
                                #                       box_head_classification=self.box_head_classification)

                                f += 1

                        if f == 0:
                            for id in self.db_gt_inactive[det_gt_id[i]]:
                                #self.exclude_from_others.append(id)
                                #logger.debug("exlude {}".format(id))
                                logger.debug("track not inactive frames anymore")
                                self.missed_reID_patience += 1
                    else:
                        logger.debug("correctly initialized new track")
                        self.correct_no_reID += 1
                        if max[i]!=old_max[i]:
                            self.correct_no_reID_iou += 1

        #handle case if there are 2 detections with high score for 1 inactive track
        for z, k in inactive_to_det.items():
            if len(k)>1:
                logger.debug('delete detections with lower score')
                det_scores = []
                for j in k:
                    det_scores.append(det_index_to_candidate[j][0][1])
                det_max_idx = np.asarray(det_scores).argmax()
                for o, j in enumerate(k):
                    if o != det_max_idx:
                        del det_index_to_candidate[j]
                        inactive_to_det[z].remove(j)
                        pass


        for det_index, candidates in det_index_to_candidate.items():
            candidate = candidates[0]
            inactive_track = candidate[0]
            # get the position of the inactive track in inactive_tracks
            # if just one track, position "is 1" because 0 is unknown background person
            # important for check in next if statement
            inactive_id_in_list = self.inactive_tracks.index(inactive_track)

            if len(inactive_to_det[inactive_id_in_list]) == 1:
                # check if GT id fits
                if inactive_track.gt_id != det_gt_id[det_index]:
                    #self.exclude_from_others.append(inactive_track.id)
                    #logger.debug("exclude reID track {} from others".format(inactive_track.id))
                    logger.debug('!!!!! wrong reID of person {}, reID GT ID {}, det GT ID {}'.format(inactive_track.id,
                                                                                        inactive_track.gt_id,
                                                                                        det_gt_id[det_index]))
                    if det_gt_id[det_index] != -1 and inactive_track.gt_id != None:
                        # just in this case i can be sure the reID was a wrong person
                        self.wrong_reID += 1
                    #print("exclude reID track {} from others".format(inactive_track.id))

                        # attach classifier
                        correct_prediction = []
                        for i, t in enumerate(self.inactive_tracks):
                            if t.gt_id == det_gt_id[det_index]:
                                correct_prediction.append(i)
                                logger.debug('correct prediction is {}'.format(i))
                        if len(correct_prediction)==0:
                            logger.debug('not possible to do correction prediction, GT ID not in inactive tracks')
                        else:
                            logger.debug('correct prediction saved')
                            # inactive_track.add_classifier(
                            #                             box_head_classification=self.box_head_classification,
                            #                             box_predictor_classification=self.box_predictor_classification,
                            #                             wrong_gt_id=det_gt_id[det_index],
                            #                             correct_prediction=correct_prediction)

                else:
                    logger.debug('correct reID of person {}'.format(inactive_track.id))
                    self.correct_reID += 1
                # make sure just 1 new detection per inactive track
                self.tracks.append(inactive_track)
                logger.debug(
                    f"\n**************   Reidying track {inactive_track.id} in frame {frame} with score {candidate[1]}")
                logger.debug(' - it was trained on inactive tracks {}'.format([t.id for t in self.inactive_tracks]))
                self.num_reids += 1
                self.inactive_count_succesfull_reID.append(inactive_track.count_inactive)

                if inactive_track.id in self.killed_this_step:
                    self.count_killed_this_step_reid += 1
                # print('\n track {} was killed and reid in frame {}'.format(inactive_track.id, self.im_index))

                # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
                if frame == 4200:
                    inactive_track.add_classifier(self.box_predictor_classification, self.box_head_classification)

                # reset inactive track
                inactive_track.count_inactive = 0
                inactive_track.pos = new_det_pos[det_index].view(1, -1)
                inactive_track.reset_last_pos()
                inactive_track.skipped_for_train = 0

                if self.finetuning_config['reset_dataset']:
                    inactive_track.frames_since_active = 1
                    inactive_track.training_set = IndividualDataset(inactive_track.id,
                                                                    self.finetuning_config['keep_frames'],
                                                                    self.finetuning_config['data_augmentation'],
                                                                    self.finetuning_config['flip_p'])

                assigned.append(det_index)
                remove_inactive.append(inactive_track)
            else:
                logger.debug('\nerror, {} new det for 1 inactive track ID {}'.format(len(inactive_to_det[inactive_id_in_list]),
                                                                              inactive_track.id))
                logger.debug(' - it was trained on inactive tracks {}'.format([t.id for t in self.inactive_tracks]))

        if len(remove_inactive) > 0:
            # do batched roi pooling
            box_roi_pool = self.obj_detect.roi_heads.box_roi_pool
            if len(remove_inactive) == 1:
                pos = remove_inactive[0].pos
            else:
                pos = torch.cat([t.pos for t in remove_inactive], 0)

            # do augmentation in current frame
            if self.finetuning_config['data_augmentation'] > 0:
                boxes = torch.tensor([]).to(device)
                for i, track in enumerate(pos):
                    box = track
                    augmented_boxes = replicate_and_randomize_boxes(box.unsqueeze(0),
                                                                    self.finetuning_config['data_augmentation'],
                                                                    self.finetuning_config['max_displacement'])
                    augmented_boxes = clip_boxes_to_image(augmented_boxes, image.shape[-2:])
                    boxes = torch.cat((boxes, torch.cat((box.unsqueeze(0), augmented_boxes))))
            else:
                # boxes = clip_boxes(pos, image.size()[1:3])
                boxes = pos

            boxes_resized = resize_boxes(boxes, image.size()[1:3], self.obj_detect.image_size[0])
            proposals = [boxes_resized]
            with torch.no_grad():
                roi_pool_feat = box_roi_pool(self.obj_detect.fpn_features, proposals, image.size()[1:3]).to(device)
            roi_pool_per_track = roi_pool_feat.split(self.finetuning_config['data_augmentation'] + 1)

        for i, inactive_track in enumerate(remove_inactive):
            self.inactive_tracks.remove(inactive_track)
            inactive_track.update_training_set_classification(features=roi_pool_per_track[i],
                                                              pos=boxes[i + self.finetuning_config[
                                                                  'data_augmentation']].unsqueeze(0),
                                                              frame=self.im_index,
                                                              area=inactive_track.calculate_area())

        keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().to(device)
        if keep.nelement() > 0:
            new_det_pos = new_det_pos[keep]
            new_det_scores = new_det_scores[keep]
        else:
            new_det_pos = torch.zeros(0).to(device)
            new_det_scores = torch.zeros(0).to(device)

    return new_det_pos, new_det_scores, track_missed

def add_oracle(self, new_det_pos, new_det_scores, image, frame, track_missed, new_det_features=None):
    """Initializes new Track objects and saves them."""
    num_new = new_det_pos.size(0)
    old_tracks = self.get_pos()
    box_roi_pool = self.obj_detect.roi_heads.box_roi_pool
    iou = bbox_overlaps(torch.cat((new_det_pos, old_tracks)), torch.cat((new_det_pos, old_tracks)))

    # do augmentation in current frame
    if self.finetuning_config['data_augmentation'] > 0:
        boxes = torch.tensor([]).to(device)
        for i, track in enumerate(new_det_pos):
            box = track
            augmented_boxes = replicate_and_randomize_boxes(box.unsqueeze(0),
                                                            self.finetuning_config['data_augmentation'],
                                                            self.finetuning_config['max_displacement'])
            augmented_boxes = clip_boxes_to_image(augmented_boxes, image.size()[1:3])
            boxes = torch.cat((boxes, torch.cat((box.unsqueeze(0), augmented_boxes))))
    else:
        #boxes = clip_boxes(new_det_pos, image.size()[1:3])
        boxes = new_det_pos

    # do batched roi pooling
    boxes_resized = resize_boxes(boxes, image.size()[1:3], self.obj_detect.image_size[0])
    proposals = [boxes_resized]
    with torch.no_grad():
        roi_pool_feat = box_roi_pool(self.obj_detect.fpn_features, proposals, image.size()[1:3]).to(device)

    roi_pool_per_track = roi_pool_feat.split(self.finetuning_config['data_augmentation'] + 1)

    # # add GT ID for new tracks
    # for d in new_det_pos:
    #     # t = torch.arange(self.im_index-(t.training_set.features.shape[0]))
    #
    #     gt_boxes = torch.cat(list(self.gt.values()), 0).to(device)
    #     new_det_iou_GT = bbox_overlaps(d, gt_boxes).cpu().numpy()
    #     ind = np.where(new_det_iou_GT == np.max(new_det_iou_GT))[1]
    #     if len(ind) > 0:
    #         ind = ind[0]
    #         overlap = new_det_iou_GT[0, ind]
    #         if overlap >= 0.5:
    #             gt_id = list(self.gt.keys())[ind]
    #             t.gt_id = gt_id
    #             self.inactive_tracks_gt_id.append(gt_id)
    #
    #             if gt_id not in self.db_gt_inactive.keys():
    #                 self.db_gt_inactive[gt_id] = [t.id]
    #             else:
    #                 self.db_gt_inactive[gt_id].append(t.id)
    #         else:
    #             print('no GT id found for track {}, overlap {}'.format(t.id, overlap))

    for i in range(num_new):
        logger.debug('init track {}'.format(self.track_num + i))
        track = Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i,
                      new_det_features[i].view(1, -1) if new_det_features else None, self.inactive_patience, self.max_features_num,
                      self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1,
                      image.size()[1:3], self.obj_detect.image_size, self.finetuning_config["batch_size"],
                      box_roi_pool=box_roi_pool, keep_frames=self.finetuning_config['keep_frames'],
                      data_augmentation = self.finetuning_config['data_augmentation'], flip_p=self.finetuning_config['flip_p'])

        if track_missed is not None:
            if track_missed[i]==-1:
                pass
            elif len(track_missed[i])==1:
                pass
                track.add_classifier(
                    box_head_classification=self.box_head_classification,
                    box_predictor_classification=self.box_predictor_classification,
                    wrong_gt_id=-1,
                    correct_prediction=track_missed[i])
            else:
                logger.debug('problem because missed reID had {} inactive tracks'.format(len(track_missed[i])))
                track.add_classifier(
                    box_head_classification=self.box_head_classification,
                    box_predictor_classification=self.box_predictor_classification,
                    wrong_gt_id=-1,
                    correct_prediction=track_missed[i])

        self.tracks.append(track)
        if frame==13800:
            # debugging frcnn-09 frame 420 problem person wird falsch erkannt in REID , aber nur einmal
            # debugging frcnn-09 frame 138 problem person wird fälschlicherweise als ID4
            track.add_classifier(self.box_predictor_classification, self.box_head_classification)
            print('\n attached classifier')
        if i in self.det_new_track_exclude:
            self.exclude_from_others.append(track.id)
            print("exclude newly init track {} from others".format(track.id))
        #other_pedestrians_bboxes = torch.cat((new_det_pos[:i], new_det_pos[i + 1:], old_tracks))
        if torch.sum(iou[i] > self.finetuning_config['train_iou_threshold']) > 1:
            print('\nSKIP SKIP SKIP beim Adden')
            self.c_skipped_for_train_iou += 1
            track.skipped_for_train += 1
            continue

        track.update_training_set_classification(features=roi_pool_per_track[i],
                                                 pos=boxes[i+self.finetuning_config['data_augmentation']].unsqueeze(0),
                                                 frame=self.im_index,
                                                 area=track.calculate_area())

    self.track_num += num_new

    # add GT ID for new tracks
    num_new = new_det_pos.size(0)
    for t in self.tracks[-num_new:]:
        # t = torch.arange(self.im_index-(t.training_set.features.shape[0]))

        gt_boxes = torch.cat(list(self.gt.values()), 0).to(device)
        new_det_iou_GT = bbox_overlaps(t.pos, gt_boxes).cpu().numpy()
        ind = np.where(new_det_iou_GT == np.max(new_det_iou_GT))[1]
        if len(ind) > 0:
            ind = ind[0]
            overlap = new_det_iou_GT[0, ind]
            if overlap >= 0.2:
                gt_id = list(self.gt.keys())[ind]
                t.gt_id = gt_id

            else:
                logger.debug('ADD no GT id found for track {}, overlap {}'.format(t.id, overlap))

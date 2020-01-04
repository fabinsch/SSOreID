import torch
from torchvision.ops.boxes import box_iou


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_random_scaling_displacement(batch_size, max_shift_x):
    x1_displacement = torch.empty(size=(batch_size, 1)).uniform_(-max_shift_x, max_shift_x).to(device)
    y1_displacement = torch.empty(size=(batch_size, 1)).uniform_(-max_shift_x, max_shift_x).to(device)
    x2_displacement = torch.empty(size=(batch_size, 1)).uniform_(-max_shift_x, max_shift_x).to(device)
    y2_displacement = torch.empty(size=(batch_size, 1)).uniform_(-max_shift_x, max_shift_x).to(device)

    return (x1_displacement, y1_displacement, x2_displacement, y2_displacement)

def apply_random_factors(gt_pos, random_factors):
    batch_size = random_factors[0].size()[0]
    training_boxes_xywh = gt_pos.repeat(batch_size, 1)
    training_boxes_xywh[:, 0:1] = training_boxes_xywh[:, 0:1] + random_factors[0]
    training_boxes_xywh[:, 1:2] = training_boxes_xywh[:, 1:2] + random_factors[1]
    training_boxes_xywh[:, 2:3] = training_boxes_xywh[:, 2:3] + random_factors[2]
    training_boxes_xywh[:, 3:4] = training_boxes_xywh[:, 3:4] + random_factors[3]

    return training_boxes_xywh

def replicate_and_randomize_boxes(gt_pos, batch_size, max_displacement=0.2, seed=1000):
    torch.manual_seed(seed)
    smallest_edge = min(abs(gt_pos[0,0]-gt_pos[0,2]), abs(gt_pos[0,1]-gt_pos[0,3]))
    factors = get_random_scaling_displacement(batch_size, max_shift_x=smallest_edge * max_displacement)
    training_boxes = apply_random_factors(gt_pos, factors)
    return training_boxes.to(device)
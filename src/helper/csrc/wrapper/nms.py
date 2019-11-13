# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from helper.csrc import _C

def nms(detections_and_scores, nms_threshold):
    detections = detections_and_scores[:, 0:4]
    scores = detections_and_scores[:, -1]
    return _C.nms(detections, scores, nms_threshold)
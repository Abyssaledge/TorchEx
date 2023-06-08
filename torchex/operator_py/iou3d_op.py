import torch

import iou3d_cuda

# to be released
# try:
#     import weighted_nms_ext
# except Exception as e:
#     print(f'Error {e} when import weighted_nms.')


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(),
                                 ans_iou)

    return ans_iou

def boxes_iou_bev_1to1(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (N, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (N, ).
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    ans_iou = boxes_a.new_zeros(boxes_a.shape[0])

    iou3d_cuda.boxes_iou_bev_1to1_gpu(boxes_a.contiguous(), boxes_b.contiguous(),
                                 ans_iou)

    return ans_iou

def boxes_overlap_1to1(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (N, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (N, ).
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    ans_overlap = boxes_a.new_zeros(boxes_a.shape[0])

    iou3d_cuda.boxes_overlap_1to1_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_overlap)

    return ans_overlap

def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep

def weighted_nms(boxes, data2merge, scores, thresh, merge_thresh, pre_maxsize=None, post_max_size=None):
    """Weighted NMS function with gpu implementation.
       Modification from the cpu version in https://github.com/TuSimple/RangeDet

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        data2merge (torch.Tensor): Input data with the shape of [N, C], corresponding to boxes. 
            If you want to merge origin boxes, just let data2merge == boxes
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Threshold.
        merge_thresh (float): boxes have IoUs with the current box higher than the threshold with weighted merged by scores.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    sorted_scores, order = scores.sort(0, descending=True)

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
        sorted_scores = sorted_scores[:pre_maxsize]

    boxes = boxes[order].contiguous()
    data2merge = data2merge[order].contiguous()
    data2merge_score = torch.cat([data2merge, sorted_scores[:, None]], 1).contiguous()
    output = torch.zeros_like(data2merge_score)
    count = torch.zeros(boxes.size(0), dtype=torch.long, device=boxes.device)

    assert data2merge_score.dim() == 2

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = weighted_nms_ext.wnms_gpu(boxes, data2merge_score, output, keep, count, thresh, merge_thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()

    assert output[num_out:, :].sum() == 0
    assert (count[:num_out] > 0).all()
    count = count[:num_out]
    output = output[:num_out, :]

    if post_max_size is not None:
        keep = keep[:post_max_size]
        output = output[:post_max_size]
        count = count[:post_max_size]
    return keep, output, count


def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh,
                                        boxes.device.index)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()

def nms_mixed_gpu(boxes, scores, batch_id, thresh, normal=False):
    """Nms function with gpu implementation for mixed batches.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        batch_id (torch.Tensor): The batch index of each box with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]
    batch_id = batch_id.int()
    batch_id = batch_id[order].contiguous()
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_mixed_gpu(boxes, keep, batch_id, thresh, normal)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    return keep


def aug_nms_gpu(boxes, scores, labels, thresh, normal=False):
    """Augmented-nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        labels (torch.Tensor): The label of each box with the shape of [N].
        thresh (int): Threshold.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]
    labels = labels.int()
    labels = labels[order].contiguous()
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.aug_nms_gpu(boxes, keep, labels, thresh, normal)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    return keep

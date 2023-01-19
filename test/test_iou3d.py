import torch
import mmcv
import numpy as np
from torchex import boxes_iou_bev, boxes_iou_bev_1to1, TorchTimer

freq = 5
seed = 0
timer = TorchTimer(freq)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0")
boxes = mmcv.load('test/test_data/iou3d_test_data.pkl').to(device)
number = boxes.shape[0]

if __name__ == '__main__':
    with torch.no_grad():
        for i in range(0,100):
            indices_1 = np.random.choice(number, number//2)
            indices_2 = np.random.choice(number, number//2)
            boxes_1 = boxes[indices_1]
            boxes_2 = boxes[indices_2]
            gt = boxes_iou_bev(boxes_1, boxes_2)[range(number//2), range(number//2)]
            with timer.timing('boxes_iou_bev_1to1'):
                pd = boxes_iou_bev_1to1(boxes_1, boxes_2)
            assert torch.isclose(gt, pd).all()


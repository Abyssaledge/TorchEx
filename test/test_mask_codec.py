import torch
import numpy as np
from torchex import mask_encoder, mask_decoder, TorchTimer

freq = 5
seed = 0
timer = TorchTimer(freq)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0")


if __name__ == '__main__':
    with torch.no_grad():
        for i in range(0,100):
            total = np.random.randint(int(1e5), int(2e5))
            mask = torch.zeros(total, dtype=bool, device=device)
            indices = np.random.choice(total, total//2)
            mask[indices] = True
            with timer.timing('Codec'):
                # assert torch.all(mask == mask_decoder(mask_encoder(mask), total))
                mask_encoder(mask)
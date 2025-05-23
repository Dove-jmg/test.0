from torch.utils.data import DataLoader
from utils import load_model
from op_calculate.utils import load_model as Lload_model
import torch
import numpy as np
import tqdm
import time

size = 150
num_samples = 500
datasets = np.load('./测试数据/taop-150-500.npy')
model, _ = load_model('./outputs/taop_150/run_20211226T072001/epoch-99.pt')
Lmodel, _ = Lload_model('./outputs/op_150/run_20211226T072001/epoch-1099.pt')
torch.manual_seed(1234)
model.cuda()
dataset = [{
                    'loc': torch.FloatTensor(datasets[i, 1:, :]).cuda(),
                    'depot': torch.FloatTensor(datasets[i, 0, :]).cuda(),
                    'prize': torch.ones(size).cuda(),
                    'max_length': torch.tensor(2.).cuda()
                } for i in range(num_samples)
        ]
Dataset = DataLoader(dataset, batch_size=1024)
batch = next(iter(Dataset))
model.eval()
Lmodel.eval()
model.set_decode_type('greedy')
start_time = time.time()
with torch.no_grad():
    length, _, tour_1, tour_2, tour_3, tour_4, tour_5, tour_6 = model(batch, Lmodel, return_pi=True)
total_time = time.time() - start_time
length = (-length.cpu().numpy()).tolist()
mean_length = np.mean(length)
output = {
    'Cost': length,
    'mean_cost': mean_length,
    'mean_time': total_time/num_samples
}
with open('taop-150-6.txt', 'w') as f:
    f.write(str(output))
    f.close()
print(length)

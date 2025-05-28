
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import math, random
from PEGPT_ablation_2 import GCNlayer,GINlayer,SAGElayer
import os
import numpy as np
import torch
from sklearn.metrics import hamming_loss,jaccard_score
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from torch_geometric.data import Batch
from torch.utils.data import random_split

class PEGPTdataset(Dataset):
    def __init__(self,pt_filename):
        super(PEGPTdataset,self).__init__()
        self.pt_filename = pt_filename
        self.pt_list = os.listdir(pt_filename)

    def get(self, item):
        graph = torch.load(self.pt_filename+'/'+self.pt_list[item])
        return graph

    def len(self):
        return len(self.pt_list)

    def name(self,item):
        return self.pt_list[item]

def flip_from_probability(p):
    return True if random.random() < p else False

def my_collate(batch_list):
    batch = Batch.from_data_list(batch_list)

def testmodel(model,device):
    print("---device---", device)
    datase = PEGPTdataset('v1_5_feature_v1') # change to your data path
    train_size = int(len(datase) * 0.6)
    valid_size = int(len(datase) * 0.2)
    test_size = int(len(datase) - train_size - valid_size)
    train_datase, valid_datase, test_datase = random_split(datase, [train_size, valid_size, test_size],)

    test_loader = DataLoader(test_datase, batch_size=1, collate_fn=my_collate)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    test_loss = 0
    test_right_count = 0
    test_count = 0
    all_targets = []
    all_preds_top1 = []
    all_preds_top5 = []
    for graph in test_loader:
        with torch.no_grad():

            x = graph.x.to(device)
            y = graph.y.to(device)
            edge_index = graph.edge_index.to(device)
            # for label in y: # test for components with rather low percentage
            #     if label != 1 and label != 2 and label != 7 and label != 64 and label != 63:
            #         if (y == label).any():
            #             first_match = (y == label).nonzero(as_tuple=True)[0][0]
            #             N2 = first_match.item()
            #             break
            #         else:
            #             print("Value not found!")
            #             N2 = random.randint(0, len(y) - 1)
            #             break
            #     else:
            #         N2 = random.randint(0, len(y) - 1)
            N2 = random.randint(0, len(y) - 1) # mask number
            tar = y[N2, 0].long()

            x[N2, :] = x[N2, :] * False

            y[N2, :] = y[N2, :] * False


            x = x.to(torch.float64)

            predict = model(x,edge_index, torch.tensor(N2).to(device),y.long())
            loss = criterion(predict.contiguous().view(-1), tar)
            top_values, top_indices = torch.topk(predict.squeeze(1), k=5)
            all_targets.append(tar.item())
            all_preds_top1.append(predict.argmax().item())
            all_preds_top5.append(top_indices.tolist())

            if tar.item() in top_indices:
                test_right_count += 1
            test_count += 1
            test_loss += loss.item()

    hammingloss = hamming_loss(all_targets, all_preds_top1)
    jscore = jaccard_score(all_targets, all_preds_top1,average='micro')
    print(f'testing Top-5 right ratio is {test_right_count / test_count}')
    print(f'testing hamming loss is {hammingloss}')
    print(f'testing jscore is {jscore}')

if __name__ == "__main__":
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = 'cuda:0'
    model = torch.load('F:\github\PEGPT_V1.0_0417_with_feature_test_50_e200_4000_0.pth',map_location=device) # change to your model path
    testmodel(model,device)

import setGPU
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
from tqdm import tqdm

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from utils import k_fold

import random
import numpy as np

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# seed_everything(12345)

test_div = 10
batch_size = 32
# test_div = 2  # 10
# batch_size = 1  # 128

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')

class Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, num_layers=2):
        super(Net, self).__init__()

        self.conv1 = GINConv(
            # Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(), Linear(dim, dim), ReLU()))
            Sequential(Linear(in_channels, dim), ReLU(), Linear(dim, dim), ReLU(), BatchNorm1d(dim)))
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    # Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(), Linear(dim, dim), ReLU()))
                    Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BatchNorm1d(dim)))
            )
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)  # -wangc
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _train(device=device, model=None, optimizer=None, train_loader=None):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def _test(device=device, model=None, loader=None, verbose=False):
    model.eval()

    total_correct = 0
    benign_correct = anomaly_correct = benign_total = anomaly_total = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int((out.argmax(-1) == data.y).sum())
        benign_correct += int((out.argmax(-1) == data.y)[data.y==0].sum())
        benign_total += len(data.y[data.y == 0])
        anomaly_correct += int((out.argmax(-1) == data.y)[data.y==1].sum())
        anomaly_total += len(data.y[data.y == 1])
    if verbose:
        print(f'\ttotal correct: {total_correct / len(loader.dataset):2.3f}, '
          # f'benign correct: {benign_correct / benign_total:2.3f}, '
          # f'anomlay correct: {anomaly_correct / anomaly_total:2.3f}'
              )
    # return total_correct / len(loader.dataset), benign_correct / benign_total, anomaly_correct / anomaly_total
    return total_correct / len(loader.dataset), -1, -1

def main(folds=10, epochs=101, verbose=False, dataset=None, batch_size=None, dim=None, num_layers=None):
    """
    
    @param folds: k folder training and testing
    @param epochs: number of epochs
    @param verbose: if output more details
    @param dataset: the TU dataset name
    @param batch_size:
    @param dim: hidden dimensions of the model
    @param num_layers: number of layers of the model
    """
    print(f'folders {folds}, epochs {epochs}, batch size {batch_size}, dim {dim}, num_layers {num_layers}')
    dataset = TUDataset(path, name=dataset)
    print('dataset:', dataset, 'dataset[0]:', dataset[0], 'dataset[-1]:', dataset[-1])
    train_acc_arr, train_benign_arr, train_anomaly_arr = [], [], []
    test_acc_arr, test_benign_arr, test_anomlay_arr = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net(dataset.num_features, 64, dataset.num_classes, num_layers=2).to(device)
    for fold, (train_idx, test_idx, val_idx) in tqdm(enumerate(zip(*k_fold(dataset, folds)))):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = Net(dataset.num_features, dim, dataset.num_classes, num_layers=num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_loss = float('inf')
        best_train_acc = best_train_benign = best_train_anomaly = 0
        best_test_acc = best_test_benign = best_test_anomaly = 0
        for epoch in range(1, epochs):
            loss = _train(device=device, model=model, optimizer=optimizer, train_loader=train_loader)
            train_acc, train_benign, train_anomaly = _test(device=device, model=model, loader=train_loader, verbose=verbose)
            test_acc, test_benign, test_anomlay = _test(device=device, model=model, loader=test_loader, verbose=verbose)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Acc: {train_acc:.3f} '
                  f'Test Acc: {test_acc:.3f}')
            if best_loss > loss:
                best_loss = loss
                best_train_acc = train_acc; best_train_benign = train_benign; best_train_anomaly = train_anomaly
                best_test_acc = test_acc; best_test_benign = test_benign; best_test_anomaly = test_anomlay
        print(f'Fold {fold}: best_loss {best_loss:.3f}, best_train_acc {best_train_acc:.3f}, best_test_acc {best_test_acc:.3f}')
        train_acc_arr.append(best_train_acc); train_benign_arr.append(best_train_benign); train_anomaly_arr.append(best_train_anomaly)
        test_acc_arr.append(best_test_acc); test_benign_arr.append(best_test_benign); test_anomlay_arr.append(best_test_anomaly)
    row_format = '{:<13}: {:.4f} ± {:.4f}\n\t [' + ' {:1.4f}'*len(train_acc_arr) + ']\n'
    print(row_format.format('train_acc', np.mean(train_acc_arr), np.std(train_acc_arr), *train_acc_arr))
    print(row_format.format('test_acc', np.mean(test_acc_arr), np.std(test_acc_arr), *test_acc_arr))

if __name__ == '__main__':
    # main(folds=10, epochs=101, verbose=False, batch_size=32, dim=64, num_layers=2, dataset='python_g_ignore_nn')
    main(folds=10, epochs=101, verbose=False, batch_size=32, dim=64, num_layers=2, dataset='python_g_ignore')
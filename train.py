import sys
sys.path.append("..")
import os
import numpy as np
import random
import pandas as pd

import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from model import IDSA, CORAL, Inter_domain_loss
from utils import get_dataset
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--source", nargs="?", default="./processed_datasets/sEMG_skku/subject_1", help="Source Datasets. ")
parser.add_argument("--target", nargs="?", default="./processed_datasets/sEMG_skku/subject_2", help="Target Datasets. ")
parser.add_argument("--batch_size", nargs="?", default=512, type=int, help="Batch size. ")
parser.add_argument("--hidden_size", nargs="?", default=128, type=int, help="hidden_size. ")
parser.add_argument("--epoch", nargs="?", default=300, type=int, help="number of epochs. ")
parser.add_argument("--num_layers", nargs="?", default=2, type=int, help="num_layers. ")
parser.add_argument("--device", nargs="?", default=0, type=int, help="gpu_num. ")
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training. (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay for training. (default: 1e-05)')
parser.add_argument('--mode', default='linear', help='mha or linear')
parser.add_argument("--seed", nargs="?", default=0, type=int, help="gpu_num. ")
parser.add_argument('--coral', type=float, default=0., help='weight value for coral loss')
parser.add_argument('--inter', type=float, default=0., help='weight value for inter-domain spatial transportation loss')
parser.add_argument('--pseudo', type=float, default=0., help='weight value for cpseudo-labeling loss')
parser.add_argument("--warmup", nargs="?", default=50, type=int, help="Warmup epochs before applying pseudo labeling loss ")
parser.add_argument('--permutation', type= bool, default= False, help='for permutation experiment')
parser.add_argument('--permutation_info', type= list, default= [2, 5], help='the sensor index that switched')
parser.add_argument('--ema', type= bool, default= True, help='apply EMA when updating target')


args = parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

set_seed(args.seed)

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

source_train_loader, source_val_loader, source_test_loader, num_labels, input_feat_dim, channel_num = get_dataset(args.source, device, args.batch_size)
target_train_loader, target_val_loader, target_test_loader, num_labels, input_feat_dim, channel_num = get_dataset(args.target, device, args.batch_size, args.permutation, args.permutation_info)

target_train_iterator = iter(target_train_loader)
# Model

model = IDSA(input_feat_dim, args.hidden_size, args.num_layers, num_labels, channel_num, mode=args.mode)
model = model.to(device)

# # model = TEMP(128, 18)
# model = model.to(device)

# Loss and optimizer
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# criterion = FocalLoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

coral = CORAL()
coral = coral.to(device)
inter = Inter_domain_loss()
inter = inter.to(device)
# Train the model
total_step = len(source_train_loader)
for epoch in range(args.epoch):
    model.train()
    train_correct = 0
    train_total = 0

    train_tgt_correct = 0
    train_tgt_total = 0
    for i, (timeseries, labels) in enumerate(source_train_loader):
        try:
            target_timeseries, target_labels = next(target_train_iterator)
        except:
            target_train_iterator = iter(target_train_loader)
            target_timeseries, target_labels = next(target_train_iterator)

        timeseries = timeseries
        labels = labels
        
        # Forward pass
        outputs, st_out_src, _, _, _ = model(timeseries)
        outputs_tgt, st_out_tgt, inter_graph, source_embed_w, target_embed_w = model(target_timeseries, target=True)
        

        # loss = criterion(outputs, labels)
        idx = (labels != 100)
        loss = criterion(outputs[idx], labels[idx])
        loss += args.coral * coral(st_out_src, st_out_tgt)
        loss += args.inter * inter(timeseries, target_timeseries, inter_graph, source_embed_w, target_embed_w)
        _, predicted = torch.max(outputs[idx].data, 1)
        train_total += labels[idx].size(0)
        train_correct += (predicted == labels[idx]).sum().item()
        
        # checking target training accuracy
        _, predicted_tgt = torch.max(outputs_tgt.data, 1)
        high_confidence_mask = (outputs_tgt.max(dim=1)[0] > 0.95)
        train_tgt_correct += (predicted_tgt[high_confidence_mask] == target_labels[high_confidence_mask]).sum().item()
        train_tgt_total += high_confidence_mask.sum().item()

        if epoch >= args.warmup: 
            loss += args.pseudo * criterion(outputs_tgt[high_confidence_mask], predicted_tgt[high_confidence_mask])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.ema:
            model.EMA(alpha=0.9)
        
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        
        # Loop through the test data
        for timeseries, labels in target_test_loader:
            # Forward pass
            outputs, st_out, _, _, _ = model(timeseries, target=True)
            idx = (labels != 100)
            _, predicted = torch.max(outputs[idx].data, 1)
            
            # Update overall accuracy metrics
            total += labels[idx].size(0)
            correct += (predicted == labels[idx]).sum().item()

        # Compute and display overall accuracy
        overall_accuracy = 100 * correct / total
        print("Epoch - {} Overall Accuracy: {:.2f}%".format(epoch, overall_accuracy))
        print('Test Accuracy of the model on the target dataset: {} %'.format(100 * correct / total), train_correct / train_total, train_tgt_correct / (train_tgt_total+1))
            
# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for timeseries, labels in source_test_loader:
        timeseries = timeseries
        # timeseries[:, [2, 3]] = timeseries[:, [3, 2]]
        labels = labels
        outputs, st_out, _, _, _ = model(timeseries, target=True)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the source dataset: {} %'.format(100 * correct / total))

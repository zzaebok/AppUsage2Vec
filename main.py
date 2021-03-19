import os
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from AppUsage2VecDataset import AppUsage2VecDataset
from AppUsage2Vec import AppUsage2Vec

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of AppUsage2Vec model")
    
    parser.add_argument('--epoch', type=int, default=10, help="The number of epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="The size of batch")
    parser.add_argument('--dim', type=int, default=64, help="The embedding size of users and apps")
    parser.add_argument('--seq_length', type=int, default=4, help="The length of previously used app sequence")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in DNN")
    parser.add_argument('--alpha', type=float, default=0.1, help="Discount oefficient for loss function")
    parser.add_argument('--topk', type=float, default=5, help="Topk for loss function")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--seed', type=int, default=2021, help="Random seed")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    # data load
    train_dataset = AppUsage2VecDataset(mode='train')
    test_dataset = AppUsage2VecDataset(mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    num_users = len(open(os.path.join('data', 'user2id.txt'), 'r').readlines())
    num_apps = len(open(os.path.join('data', 'app2id.txt'), 'r').readlines())
    
    # model & optimizer
    model = AppUsage2Vec(num_users, num_apps, args.dim, args.seq_length, args.num_layers, args.alpha, args.topk, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # train & evaluation
    total_loss = 0
    itr = 1
    p_itr = 500
    best_acc = 0
    Ks = [1,5,10]
    acc_history = [[0,0,0]]
    best_acc = 0
    
    for e in range(args.epoch):
        
        model.train()
        for i, (data, targets) in enumerate(train_loader):
            users = data[0].to(device)
            time_vecs = data[1].to(device)
            app_seqs = data[2].to(device)
            time_seqs = data[3].to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            loss = model(users, time_vecs, app_seqs, time_seqs, targets, 'train')
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if itr % p_itr == 0:
                print("[TRAIN] Epoch: {} / Iter: {} Loss - {}".format(e+1, itr, total_loss/p_itr))
                total_loss = 0
            itr += 1
            
        model.eval()
        corrects = [0, 0, 0]
        with torch.no_grad():
            for i, (data, targets) in enumerate(test_loader):
                users = data[0].to(device)
                time_vecs = data[1].to(device)
                app_seqs = data[2].to(device)
                time_seqs = data[3].to(device)
                targets = targets.to(device)
                
                scores = model(users, time_vecs, app_seqs, time_seqs, targets, 'predict')
                for idx, k in enumerate(Ks):
                    correct = torch.sum(torch.eq(torch.topk(scores, dim=1, k=k).indices, targets)).item()
                    corrects[idx] += correct
        accs = [x/len(test_dataset) for x in corrects]
        acc_history.append(accs)
        print("[EVALUATION] Epoch: {} - Acc: {:.5f} / {:.5f} / {:.5f}".format(e+1, accs[0], accs[1], accs[2]))
        
        if accs[2] > best_acc:
            best_acc = accs[2]
    
    print("BEST ACC@10: {}".format(best_acc))
                
    # visualization
    fig = plt.figure()
    acc_history = list(map(list, zip(*acc_history)))
    for idx, k in enumerate(Ks):
        plt.plot(acc_history[idx], label='acc@{}'.format(k))
    plt.xlabel('epoch')
    plt.ylabel('accuracy@k')
    plt.legend(loc='upper left')
    plt.xticks(np.arange(args.epoch+1))
    plt.ylim(0, 1)
    fig.savefig('result.png')
    

if __name__ == "__main__":
    main()
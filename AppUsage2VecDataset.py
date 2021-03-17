import ast
import pandas as pd
from torch.utils.data import Dataset
import torch

class AppUsage2VecDataset(Dataset):
    """AppUsage2Vec Dataset
    
    Args:
        mode(str): which dataset will you make, 'train' or 'test'
    """
    
    def __init__(self, mode):
        if mode == 'train':
            self.df = pd.read_csv('data/train.txt', sep='\t')
        else:
            self.df = pd.read_csv('data/test.txt', sep='\t')
        
        self.df['app_seq'] = self.df['app_seq'].apply(ast.literal_eval)
        self.df['time_seq'] = self.df['time_seq'].apply(ast.literal_eval)
    
    def __len__(self):
        return  len(self.df)
    
    def __getitem__(self, idx):
        user = self.df.iloc[idx]['user']
        time = self.df.iloc[idx]['time']
        target = self.df.iloc[idx]['app']
        app_seq = self.df.iloc[idx]['app_seq']
        time_seq = self.df.iloc[idx]['time_seq']
        time_vector = torch.zeros(31)
        
        # time vector one of 7 dim / one of 24 dim
        time_vector[list(map(int, time.split('_')))] = 1
        return torch.LongTensor([user]), time_vector, torch.LongTensor(app_seq), torch.Tensor(time_seq), torch.LongTensor([target])
        
        
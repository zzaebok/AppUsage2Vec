import ast
import pandas as pd
from torch.utils.data import Dataset


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
    
    def __len__(self):
        return  len(self.df)
    
    def __getitem__(self, idx):
        user = self.df.iloc[idx]['user']
        time = self.df.iloc[idx]['time']
        time_vector = torch.zeros(31)
        time_vector[time.split('_')] = 1
        
        
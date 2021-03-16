import torch
import torch.nn as nn

class AppUsage2Vec(nn.module):
    def __init__(self, n_users, n_apps, dim, seq_length, n_layers):
        super(AppUsage2Vec).__init__()
        
        self.user_emb = nn.Embedding(n_users, dim)
        self.app_emb = nn.Embedding(n_apps, dim)
        self.seq_length = seq_length
        
        self.attn_weight = nn.Linear((dim+1)*seq_length, seq_length)
        
        self.user_dnn = nn.ModuleList([nn.Linear(dim, dim) for i in range(n_layers)])
        self.app_dnn = nn.ModuleList([nn.Linear(dim, dim) for i in range(n_layers)])
        
    def forward(self):
        # make time vector / Eq.(11)
        
        # attach time difference to each app embedding in the sequence / Eq.(12)
        
        # get sequence vector / Eq.(6)
        
        # dual dnn / Eq.(7)(8)
        
        # hadamard product / Eq.(10)
        
        # concat hidden vector and time vector / Eq.(13)
        
        # softmax / Eq.(4)
        pass
        
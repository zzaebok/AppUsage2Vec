import torch
import torch.nn as nn
import torch.nn.functional as F

class AppUsage2Vec(nn.Module):
    def __init__(self, n_users, n_apps, dim, seq_length, n_layers, alpha, k, device):
        super(AppUsage2Vec, self).__init__()
        
        self.user_emb = nn.Embedding(n_users, dim)
        self.app_emb = nn.Embedding(n_apps, dim)
        self.seq_length = seq_length
        self.alpha = alpha
        self.k = k
        self.device = device
        
        self.attn = nn.Linear(seq_length * (dim+1), seq_length)
        
        self.n_layers = n_layers
        self.user_dnn = nn.ModuleList([nn.Linear(dim, dim) for i in range(n_layers)])
        self.app_dnn = nn.ModuleList([nn.Linear(dim, dim) for i in range(n_layers)])
        
        self.classifier = nn.Linear(dim+31, n_apps)
        
    def forward(self, users, time_vecs, app_seqs, time_seqs, targets, mode):
        # users [batch_size, 1]
        # time_vecs [batch_size, 31]
        # app_seqs [batch_size, seq_length]
        # time_seqs [batch_size, seq_length]
        # targets [batch_size, 1]
        
        # attach time difference to each app embedding in the sequence / Eq.(12)
        app_seqs_emb = self.app_emb(app_seqs)  # [batch_size, seq_length, dim]
        time_seqs = time_seqs.unsqueeze(2)  # [batch_size, seq_length, 1]
        app_seqs_time = torch.cat([app_seqs_emb, time_seqs], dim=2)  # [batch_size, seq_length, dim+1]
        
        app_seqs_flat = app_seqs_time.view(app_seqs_time.size(0), -1)  # [batch_size, seq_length * (dim+1)]
        
        # get sequence vector / Eq.(6)
        H_v = torch.tanh(self.attn(app_seqs_flat))  # [batch_size, seq_length]
        weights = F.normalize(H_v, p=1, dim=1)  # [batch_size, seq_length]
        
        # [batch_size, dim, seq_length] * [batch_size, seq_length, 1] = [batch_size, dim]
        seq_vector = torch.bmm(app_seqs_emb.permute(0, 2, 1), weights.unsqueeze(2)).squeeze(2)
        
        # dual dnn / Eq.(7)(8)
        user_vector = self.user_emb(users).squeeze(1)  # [batch_size, dim]
        for i in range(self.n_layers):
            user_vector = self.user_dnn[i](user_vector)
            user_vector = torch.tanh(user_vector)
            seq_vector = self.app_dnn[i](seq_vector)
            seq_vector = torch.tanh(seq_vector)
        
        # hadamard product / Eq.(10)
        combination = torch.mul(user_vector, seq_vector)  # [batch_size, dim]
        
        # concat hidden vector and time vector / Eq.(13)
        combination = torch.cat([combination, time_vecs], dim=1) # [batch_size, dim+31]
        
        # softmax / Eq.(4)
        scores = F.softmax(self.classifier(combination), dim=1)  # [batch_size, n_apps]
        
        if mode == 'predict':
            return scores  # [batch_size, n_apps]
        else:
            preds = torch.topk(scores, dim=1, k=self.k).indices  # [batch_size, k]
            indicator = torch.sum(torch.eq(preds, targets), dim=1)  # [batch_size]
            coefficient = torch.pow(torch.Tensor([self.alpha] * indicator.size(0)).to(self.device), indicator) # [batch_size]
            p = scores[torch.arange(scores.size(0)), targets.view(-1)] # [batch_size]
            plogp = p * torch.log(p) # [batch_size]
            loss = (-1.0) * torch.mean(coefficient * plogp)
            return loss
            
        
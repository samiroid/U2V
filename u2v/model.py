import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from numpy.random import RandomState
from tadat.core.helpers import colstr
import math

SHUFFLE_SEED=10

class User2Vec(nn.Module):

    def __init__(self, user_id, emb_dimension, outpath, margin=10, initial_lr=0.1, 
                 epochs=10, batch_size=None, device=None):
        super(User2Vec, self).__init__()     
        if not device:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        print("[device: {}]".format(self.device))      
        self.outpath = outpath        

        self.user_id = user_id
        self.emb_dimension = emb_dimension
        #user embedding matrix
        self.U = nn.Embedding(1, self.emb_dimension)
        initrange = 1.0 / self.emb_dimension
        #init weights
        init.uniform_(self.U.weight.data, -initrange, initrange)
       
        
        self.margin = margin
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.epochs = epochs
        

    def forward(self, idxs):
        #content embeddings
        emb_pos = self.positive_samples[idxs]
        emb_neg = self.negative_samples[idxs]
         #user embedding
        emb_user = self.U(torch.tensor([0], device=self.device))                

        #prediction
        logits = emb_pos @ emb_user.T        
        neg_logits = emb_neg @ emb_user.T
        #loss: max(0, margin - pos + neg)             
        zero_tensor = torch.tensor([0], device=self.device).float().expand_as(logits)        
        loss = torch.max(zero_tensor, (self.margin - logits + neg_logits))
        return loss.mean()
    
    def doc_proba(self, docs):        
        #user embedding
        emb_user = self.U(torch.tensor([0], device=self.device))                
        #conditonal word likelihood 
        logits = docs @ emb_user.T        
        probs = torch.sigmoid(logits.squeeze())        
        return torch.mean(probs)        

    def fit(self, X_positive, X_negative, X_val):        
        self.to(self.device)
        assert self.emb_dimension == X_positive.shape[-1]
        N = X_positive.shape[0]
        V = X_val.shape[0]        
        self.positive_samples = X_positive.to(self.device)
        self.negative_samples = X_negative.to(self.device)
        self.validation = X_val.to(self.device)

        rng = RandomState(SHUFFLE_SEED)       
        optimizer = optim.Adam(self.parameters(), lr=self.initial_lr)       

        val_prob=0
        best_val_prob=0    
        n_val_drops=0   
        MAX_VAL_DROPS=5
        loss_margin = 0.005      
        if self.batch_size:
            n_batches = math.ceil(N/self.batch_size)
        else:
            n_batches = 1
            self.batch_size = N
        
        train_idx = np.arange(N).reshape(1,-1)
        

        for e in range(self.epochs):        
            train_idx = rng.permutation(N)            
            running_loss = 0.0        
            for j in range(n_batches):               
                #get batches 
                batch_idx = train_idx[j*self.batch_size:(j+1)*self.batch_size]                   
                # from ipdb import set_trace; set_trace()
                optimizer.zero_grad()
                loss = self.forward(batch_idx)
                # print(loss.item())
                loss.backward()
                optimizer.step()                
                running_loss += loss.item() 
            avg_loss = round(running_loss/N,4)
            val_prob = 0
            val_prob = round(self.doc_proba(self.validation).item(), 4)
            # val_prob = val_prob
            status_msg = "epoch: {} | loss: {} | val avg prob: {} ".format(e, avg_loss, val_prob)
            if val_prob > best_val_prob:    
                n_val_drops=0            
                best_val_prob = val_prob
                self.save_embedding()                
                status_msg = colstr(status_msg, "green")
            elif val_prob < (best_val_prob - loss_margin):                
                n_val_drops+=1
                if n_val_drops == MAX_VAL_DROPS:
                    print("[early stopping: {} epochs]".format(e))
                    break
                status_msg = colstr(status_msg, "red")            
            print(status_msg)                

    def save_embedding(self): 
        with open(self.outpath+self.user_id+".txt","w") as fo:
            embedding = self.U.weight.cpu().data.numpy()[0]                
            fo.write('%d %d\n' % (1, self.emb_dimension))            
            e = ' '.join(map(lambda x: str(x), embedding))
            fo.write('%s %s\n' % (self.user_id, e))

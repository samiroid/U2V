import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from numpy.random import RandomState
from tadat.core.helpers import colstr
import math
import sys
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
        self.margin = margin
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.epochs = epochs
                
        self.emb_dimension = emb_dimension
        #user embedding matrix
        self.U = nn.Embedding(1, self.emb_dimension)
        initrange = 1.0 / self.emb_dimension
        #init weights
        init.uniform_(self.U.weight.data, -initrange, initrange)

    def forward(self, emb_pos, emb_neg):
        #content embeddings
        # emb_pos = self.positive_samples[idxs]
        # emb_neg = self.negative_samples[idxs]
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
        with torch.no_grad():
            #user embedding
            emb_user = self.U(torch.tensor([0], device=self.device))                
            #conditonal word likelihood 
            logits = docs @ emb_user.T        
            probs = torch.sigmoid(logits.squeeze())        
        return torch.mean(probs)        

    def fit(self, X_positive, X_negative, X_val):        
        batches = X_positive.files
        assert self.emb_dimension == X_positive[batches[0]].shape[1]
        N  = X_positive[batches[0]].shape[0] * len(batches)
        n_val = X_val.shape[0]
        print("{} | tr: {} | ts: {}".format(self.user_id, N, n_val))
        
        #to device
        self.to(self.device)
        self.validation = torch.from_numpy(X_val.astype(np.float32)).to(self.device)     
        

        optimizer = optim.Adam(self.parameters(), lr=self.initial_lr)       
        rng = RandomState(SHUFFLE_SEED)       
        val_prob=0
        best_val_prob=0    
        n_val_drops=0   
        MAX_VAL_DROPS=5
        loss_margin = 0.005      
        # train_idx = np.arange(N).reshape(1,-1)        
        for e in range(self.epochs):        
            # train_idx = rng.permutation(N)            
            running_loss = 0.0       
            rng.shuffle(batches)
            for batch in batches:
                # sys.stdout.write(f"\r> batch {batch}")
                # sys.stdout.flush()          
                #get batches 
                pos_batch = torch.from_numpy(X_positive[batch].astype(np.float32)).to(self.device)     
                neg_batch = torch.from_numpy(X_negative[batch].astype(np.float32)).to(self.device)     
                # pos_batch = pos_batch
                # neg_batch = neg_batch
                # batch_idx = train_idx[j*self.batch_size:(j+1)*self.batch_size]                   
                optimizer.zero_grad()
                loss = self.forward(pos_batch, neg_batch)                
                loss.backward()
                optimizer.step()                
                running_loss += loss.item() 
            avg_loss = round(running_loss/N,3)            
            val_prob = round(self.doc_proba(self.validation).item(), 3)
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

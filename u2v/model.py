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
import time
SHUFFLE_SEED=10

class User2Vec(nn.Module):

    def __init__(self, user_id, emb_dimension, outpath, margin=10, initial_lr=0.1, 
                  validation_split=0.8, epochs=10, batch_size=None, device=None):
        super(User2Vec, self).__init__()     
        if not device:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert validation_split >= 0 and validation_split <=1
        self.device = device
        print("[device: {}]".format(self.device))      
        self.outpath = outpath     
        self.user_id = user_id
        self.margin = margin
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.training_split = 1-validation_split
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

    def get_batch(self, data, windows):
        batch = []
        for w in windows:
            batch.append(data[w])
        batch = np.vstack(batch)   
        batch = torch.from_numpy(batch.astype(np.float32))
        return batch

    def fit(self, X_positive, X_negative):        
        st = time.time()
        #TODO: do I need this? 
        #to device
        self.to(self.device)

        #positives and negatives have same file names
        windows = X_positive.files
        assert self.emb_dimension == X_positive[windows[0]].shape[1]
        #split for validation
        n_val = 0
        split_idx = math.floor(len(windows)*self.training_split)
        train_windows = windows[:split_idx]
        if self.training_split < 1:
            val_windows = windows[split_idx:]
            X_val = self.get_batch(X_positive, val_windows)
            n_val = X_val.shape[0]
            validation = X_val.to(self.device)    
        n_windows = len(train_windows)
        N  = X_positive[train_windows[0]].shape[0] * n_windows
        
        print("{} | tr: {} ({}) | val: {}".format(self.user_id, N, n_windows, n_val))        
        optimizer = optim.Adam(self.parameters(), lr=self.initial_lr)       
        rng = RandomState(SHUFFLE_SEED)       
        val_prob=0
        best_val_prob=0    
        n_val_drops=0   
        MAX_VAL_DROPS=5
        loss_margin = 0.005         
        n_batches = math.ceil(n_windows/self.batch_size)
        for e in range(self.epochs):                    
            running_loss = 0.0       
            rng.shuffle(train_windows)
            for i in range(n_batches):                
                #get batches 
                batch = train_windows[i*self.batch_size:(i+1)*self.batch_size]                
                pos_batch = self.get_batch(X_positive,batch).to(self.device)
                neg_batch = self.get_batch(X_negative,batch).to(self.device)                

                optimizer.zero_grad()
                loss = self.forward(pos_batch, neg_batch)                
                loss.backward()
                optimizer.step()                
                running_loss += loss.item() 
            avg_loss = round(running_loss/N,4)       
            if self.training_split < 1:     
                val_prob = round(self.doc_proba(validation).item(), 4)
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
            else:
                status_msg = "epoch: {} | loss: {}".format(e, avg_loss)
            print(status_msg)                
        ft = time.time()
        et = ft - st
        print(f"time: {round(et,3)}")
        return best_val_prob

    # def fit(self, X_positive, X_negative, X_val):        
    #     st = time.time()
    #     batches = X_positive.files
    #     assert self.emb_dimension == X_positive[batches[0]].shape[1]
    #     n_batches = len(batches)
    #     N  = X_positive[batches[0]].shape[0] * n_batches
    #     n_val = X_val.shape[0]
    #     print("{} | tr: {} ({}) | val: {}".format(self.user_id, N, n_batches,  n_val))        
    #     #to device
    #     self.to(self.device)
    #     validation = torch.from_numpy(X_val.astype(np.float32)).to(self.device)   
    #     optimizer = optim.Adam(self.parameters(), lr=self.initial_lr)       
    #     rng = RandomState(SHUFFLE_SEED)       
    #     val_prob=0
    #     best_val_prob=0    
    #     n_val_drops=0   
    #     MAX_VAL_DROPS=5
    #     loss_margin = 0.005              
    #     for e in range(self.epochs):                    
    #         running_loss = 0.0       
    #         rng.shuffle(batches)
    #         for batch in batches:                
    #             #get batches 
    #             pos_batch = torch.from_numpy(X_positive[batch].astype(np.float32)).to(self.device)     
    #             neg_batch = torch.from_numpy(X_negative[batch].astype(np.float32)).to(self.device)                     
    #             # pos_batch = self.get_batch(X_positive,[batch]).to(self.device)
    #             # neg_batch = self.get_batch(X_negative,[batch]).to(self.device)
    #             # neg_batch = torch.from_numpy(X_negative[batch].astype(np.float32)).to(self.device)                     

    #             optimizer.zero_grad()
    #             loss = self.forward(pos_batch, neg_batch)                
    #             loss.backward()
    #             optimizer.step()                
    #             running_loss += loss.item() 
    #         avg_loss = round(running_loss/N,4)            
    #         val_prob = round(self.doc_proba(validation).item(), 4)
    #         # val_prob = val_prob
    #         status_msg = "epoch: {} | loss: {} | val avg prob: {} ".format(e, avg_loss, val_prob)
    #         if val_prob > best_val_prob:    
    #             n_val_drops=0            
    #             best_val_prob = val_prob
    #             self.save_embedding()                
    #             status_msg = colstr(status_msg, "green")
    #         elif val_prob < (best_val_prob - loss_margin):                
    #             n_val_drops+=1
    #             if n_val_drops == MAX_VAL_DROPS:
    #                 print("[early stopping: {} epochs]".format(e))
    #                 break
    #             status_msg = colstr(status_msg, "red")            
    #         print(status_msg)                
    #     ft = time.time()
    #     et = ft - st
    #     print(f"time: {round(et,3)}")

    def save_embedding(self): 
        with open(self.outpath+self.user_id+".txt","w") as fo:
            embedding = self.U.weight.cpu().data.numpy()[0]                
            fo.write('%d %d\n' % (1, self.emb_dimension))            
            e = ' '.join(map(lambda x: str(x), embedding))
            fo.write('%s %s\n' % (self.user_id, e))

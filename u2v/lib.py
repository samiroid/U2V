import argparse
# import codecs
from collections import Counter
# from ipdb import set_trace
import glob
from pathlib import Path
import os
import pickle
import random
import shutil
import sys

import numpy as np
import torch
from tadat.core import embeddings
import model 
import encoders 
MIN_DOC_LEN=2
MIN_DOCS = 2

class NegativeSampler():

    def __init__(self, vocab, word_count, warp=0.75, n_neg_samples=5):
        '''
        Store count for the range of indices in the dictionary
        '''
        self.n_neg_samples = n_neg_samples
        index2word = {i:w for w,i in vocab.items()}	
        # set_trace()
        max_index = max(index2word.keys())
        counts = []
        for n in range(max_index):
            if n in index2word:
                counts.append(word_count[index2word[n]]**warp)
            else:    
                counts.append(0)
        counts = np.array(counts)
        norm_counts = counts/sum(counts)
        scaling = int(np.ceil(1./min(norm_counts[norm_counts>0])))
        scaled_counts = (norm_counts*scaling).astype(int)
        self.cumsum = scaled_counts.cumsum()   

    def sample(self, size):        
        total_size = np.prod(size)
        random_ints = np.random.randint(self.cumsum[-1], size=total_size)
        data_y_neg = np.searchsorted(self.cumsum, random_ints).astype('int32')            
        return data_y_neg.reshape(size) 

    def sample_filtered(self, exclude, size):        
        total_size = np.prod(size)
        random_ints = np.random.randint(self.cumsum[-1], size=total_size)
        data_y_neg = np.searchsorted(self.cumsum, random_ints).astype('int32')
        #filter out words that should be excluded 
        filtered = [x for x in data_y_neg.tolist() if x not in exclude][:total_size]
        data_y_neg = np.array(filtered)         
        return data_y_neg.reshape(size) 

def get_vocabulary(inpath, min_word_freq=5, max_vocab_size=None):    
    print(" > extracting vocabulary")
    word_counter = Counter()
    n_docs=0	
    # doc_lens = []
    with open(inpath) as fid:	
        for line in fid:			
            #discard first token (user id)            
            message = line.split()[1:]            
            word_counter.update(message)				
            n_docs+=1
            # doc_lens+=[len(message)]
    #keep only words that occur at least min_word_freq times
    wc = {w:c for w,c in word_counter.items() if c>min_word_freq} 
    #keep only the max_vocab_size most frequent words
    tw = sorted(wc.items(), key=lambda x:x[1],reverse=True)
    vocab = {w[0]:i for i,w in enumerate(tw[:max_vocab_size])}	

    return vocab, word_counter

def extract_word_embeddings(embeddings_path, vocab, encoding="latin-1"):    
    print(" > loading word embeddings")
    full_E = embeddings.read_embeddings(embeddings_path, vocab, encoding)
    ooevs = embeddings.get_OOEVs(full_E, vocab)
    #keep only words with pre-trained embeddings    
    for w in ooevs:
        del vocab[w]	
    vocab_redux = {w:i for i,w in enumerate(vocab.keys())}	    
    #generate the embedding matrix
    emb_size = full_E.shape[0]
    E = np.zeros((int(emb_size), len(vocab_redux)))   
    for wrd,idx in vocab_redux.items(): 
        E[:, idx] = full_E[:,vocab[wrd]]	       
    
    return E, vocab_redux

def build_data(inpath, outpath, embeddings_path, emb_encoding="latin-1", 
                min_word_freq=5, max_vocab_size=None, random_seed=123, n_neg_samples=10, 
                min_docs_user=2, reset=False):
    pkl_path=outpath+"pkl/"
    users_path=pkl_path+"users/"  
    # raise NotImplementedError
    if reset:
        shutil.rmtree(pkl_path, ignore_errors=True)
        shutil.rmtree(users_path, ignore_errors=True)

    if not os.path.exists(os.path.dirname(users_path)):
        os.makedirs(os.path.dirname(users_path))   

    vocab = None
    word_counts = None
    try:
        with open(pkl_path+"vocab.pkl", "rb") as fi:        
            vocab, word_counts = pickle.load(fi)        
            print("[found cached vocabulary]")
    except FileNotFoundError:
        pass

    if not vocab:
        #compute vocabulary
        vocab, word_counts = get_vocabulary(inpath, min_word_freq=min_word_freq,max_vocab_size=max_vocab_size)        
        vocab_len = len(vocab)
        #extract word embeddings
        E, vocab_redux = extract_word_embeddings(embeddings_path, vocab, encoding=emb_encoding)
        #vocab_redux has only words for which an embedding was found
        print("[vocab size: {} > {}]".format(vocab_len,len(vocab_redux)))
        vocab = vocab_redux
        with open(pkl_path+"word_emb.npy", "wb") as f:
            np.save(f, E)    
        with open(pkl_path+"vocab.pkl", "wb") as f:
            pickle.dump([vocab_redux, word_counts], f, pickle.HIGHEST_PROTOCOL)
    
    rng = np.random.RandomState(random_seed)    
    with open(inpath) as fi:
        #peek at the first line to get the first user
        curr_user, doc = fi.readline().replace("\"", "").replace("'","").split("\t")
        #read file from the start
        fi.seek(0,0)
        user_docs = []
        doc_lens = []
        users = []
        for line in fi:                        
            user, doc = line.replace("\"", "").replace("'","").split("\t")            
            #if we reach a new user, save the current one
            if user!= curr_user:
                if len(user_docs) >= min_docs_user:
                    max_doc_len = max(doc_lens)
                    save_user(curr_user, user_docs, max_doc_len, rng, users_path)
                    users.append(curr_user)
                else:
                    print("> IGNORED user: {}  ({})".format(user,len(user_docs)))
                #reset current user
                curr_user = user
                user_docs = []  
                doc_lens = []
            doc = doc.split(" ")            
            doc_idx = [vocab[w] for w in doc if w in vocab]		
            doc_len = len(doc_idx)	    
            if doc_len < MIN_DOC_LEN: continue
            #accumulate all texts
            user_docs.append(np.array(doc_idx, dtype=np.int32).reshape(1,-1))        
            doc_lens.append(doc_len)
        #save last user
        if len(user_docs) >= min_docs_user:
            max_doc_len = max(doc_lens)
            save_user(curr_user, user_docs, max_doc_len,  rng, users_path)
            users.append(curr_user)
        else:
            print("> IGNORED user: {}  ({})".format(user,len(user_docs)))
    

    with open(pkl_path+"users.txt","w") as fo:
        fo.write("\n".join(users))

    print("\n> building negative samples")
    build_negative_samples(pkl_path, n_neg_samples)
    build_features(pkl_path)

def save_user(user_id, docs, max_doc_len, rng, outpath, split=0.8):        
    sys.stdout.write("\r> saving user: {}  ({})".format(user_id,len(docs)))
    sys.stdout.flush()
    #shuffle the data
    shuf_idx = np.arange(len(docs))
    rng.shuffle(shuf_idx)
    docs_shuf = [docs[i] for i in shuf_idx]
    split_idx = int(len(docs)*split)
    docs_train = docs_shuf[:split_idx]
    docs_val = docs_shuf[split_idx:]
    
    with open(outpath+"temp_"+user_id, "wb") as fo:        
        pickle.dump([user_id, max_doc_len, docs_train, docs_val], fo)

def build_negative_samples(inpath, n_samples):
    with open(inpath+"vocab.pkl", "rb") as fi:        
            vocab, word_counts = pickle.load(fi)        
    sampler = NegativeSampler(vocab, word_counts, warp=0.75)
    for tmp_user_path in glob.glob(inpath+"/users/temp_*"):
        with open(tmp_user_path, "rb") as fi:
            user_id, max_doc_len, docs_train, docs_val = pickle.load(fi)
        pos_samples = []
        neg_samples = []                
        for x in docs_train:                    
            #calculate negative samples 
            neg_sample = sampler.sample((n_samples, x.shape[1]))    
            neg_samples += np.split(neg_sample, n_samples) 
            #replicate each sample to match the number of negative samples            
            for i in range(n_samples): pos_samples.append(x)
        # from ipdb import set_trace; set_trace()
        sys.stdout.write("\r> negative samples | user: {}".format(user_id))
        sys.stdout.flush()
        
        with open(inpath+"users/idx_"+user_id, "wb") as fo:        
            pickle.dump([user_id, max_doc_len, pos_samples, docs_val, neg_samples], fo)
        os.remove(tmp_user_path)

def build_features(inpath):
    with open(inpath+"vocab.pkl", "rb") as fi:        
        vocab, word_counts = pickle.load(fi)        
    with open(inpath+"word_emb.npy", "rb") as f:
        E = np.load(f)    
    for user_path in glob.glob(inpath+"/users/idx_*"):
        with open(user_path, "rb") as fi:
            user_id, max_doc_len, positive, validation, negative = pickle.load(fi)                
        X_positive = encoders.word2vec_encoder(positive, max_doc_len, "bin", E)
        X_negative = encoders.word2vec_encoder(negative, max_doc_len, "bin", E)
        X_validation = encoders.word2vec_encoder(validation, max_doc_len, "bin", E)
        with open(inpath+"/users/w2v_"+user_id+"_pos.npy", "wb") as fi:
            np.save(fi, X_positive)
        with open(inpath+"/users/w2v_"+user_id+"_neg.npy", "wb") as fi:
            np.save(fi, X_negative)
        with open(inpath+"/users/w2v_"+user_id+"_val.npy", "wb") as fi:
            np.save(fi, X_validation)

def stich_embeddings(inpath, outpath, emb_dim):
    print("[writing embeddings to {}]".format(outpath))
    with open(outpath,"w") as fo:    
        user_embeddings = list(Path(inpath).iterdir())
        fo.write("{}\t{}\n".format(len(user_embeddings), emb_dim))
        for u in user_embeddings:
            with open(u, "r") as fi:
                l = fi.readlines()[1]
            fo.write(l)

def train_model(path, encoder="w2v", epochs=20, initial_lr=0.001, margin=1, reset=False, device=None):
    txt_path = path+"/txt/"    
    if reset:
        shutil.rmtree(txt_path, ignore_errors=True)    
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path))       
    with open(path+"/pkl/users.txt") as fi:
        users = [u.replace("\n","") for u in fi.readlines()]
    
    random.shuffle(users)
    cache = set([os.path.basename(f).replace(".txt","") for f in Path(txt_path).iterdir()])
    emb_dim = None
    for user in users:    
        if user in cache:
            print("cached embedding: {}".format(user))
            continue
        user_fname = "{}/pkl/users/{}_{}_{}.npy" 
        with open(user_fname.format(path, encoder, user, "pos"), "rb") as fi:
            pos_samples = np.load(fi)
            pos_samples = torch.from_numpy(pos_samples.astype(np.float32))     
        with open(user_fname.format(path, encoder, user, "neg"), "rb") as fi:
            neg_samples = np.load(fi)
            neg_samples = torch.from_numpy(neg_samples.astype(np.float32))     
        with open(user_fname.format(path, encoder, user, "val"), "rb") as fi:
            val_samples = np.load(fi)
            val_samples = torch.from_numpy(val_samples.astype(np.float32))     
        
        print("{} | tr: {} | ts: {}".format(user, pos_samples.shape[0], val_samples.shape[0]))
        emb_dim = pos_samples.shape[-1]
        f = model.User2Vec(user, emb_dim, txt_path, margin=1, initial_lr=10, epochs=20, device=device, batch_size=100)   
        f.fit(pos_samples, neg_samples, val_samples)
        # break
    if emb_dim:
        stich_embeddings(txt_path, path+"{}_U.txt".format(encoder), emb_dim)


# def train_model(path,  epochs=20, initial_lr=0.001, margin=1, reset=False, device=None):
#     txt_path = path+"/txt/"    
#     if reset:
#         shutil.rmtree(txt_path, ignore_errors=True)
    
#     if not os.path.exists(os.path.dirname(txt_path)):
#         os.makedirs(os.path.dirname(txt_path))   
    
#     E = np.load(path+"/pkl/word_emb.npy")    
#     E = torch.from_numpy(E.astype(np.float32))     
#     user_data = list(Path(path+"/pkl/users/").iterdir())
#     random.shuffle(user_data)
#     cache = set([os.path.basename(f).replace(".txt","") for f in Path(txt_path).iterdir()])
    
#     for j, user_fname in enumerate(user_data):
#         user = os.path.basename(user_fname) 
#         if user in cache:
#             print("cached embedding: {}".format(user))
#             continue        
#         with open(user_fname, "rb") as fi:
#             user_id, pos_samples, neg_samples, val_samples = pickle.load(fi)
        
#         print("{} | tr: {} | ts: {}".format(user_id,len(pos_samples), len(val_samples)))
#         f = model.User2Vec(user_id, E.T, txt_path, margin=margin, initial_lr=initial_lr, 
#                         epochs=epochs, device=device)    
#         f.fit(pos_samples, neg_samples, val_samples)
#     stich_embeddings(txt_path, path, E.shape[0])




# def build_data(inpath, outpath, embeddings_path, emb_encoding="latin-1", 
#                 min_word_freq=5, max_vocab_size=None, random_seed=123, n_neg_samples=10, 
#                 min_docs_user=2, reset=False):
#     pkl_path=outpath+"pkl/"
#     users_path=pkl_path+"users/"  

#     if reset:
#         shutil.rmtree(pkl_path, ignore_errors=True)
#         shutil.rmtree(users_path, ignore_errors=True)

#     if not os.path.exists(os.path.dirname(users_path)):
#         os.makedirs(os.path.dirname(users_path))   

#     vocab = None
#     word_counts = None
#     try:
#         with open(pkl_path+"vocab.pkl", "rb") as fi:        
#             vocab, word_counts = pickle.load(fi)        
#             print("[found cached vocabulary]")
#     except FileNotFoundError:
#         pass

#     if not vocab:
#         #compute vocabulary
#         vocab, word_counts = get_vocabulary(inpath, min_word_freq=min_word_freq,max_vocab_size=max_vocab_size)        
#         vocab_len = len(vocab)
#         #extract word embeddings
#         E, vocab_redux = extract_word_embeddings(embeddings_path, vocab, encoding=emb_encoding)
#         #vocab_redux has only words for which an embedding was found
#         print("[vocab size: {} > {}]".format(vocab_len,len(vocab_redux)))
#         vocab = vocab_redux
#         with open(pkl_path+"word_emb.npy", "wb") as f:
#             np.save(f, E)    
#         with open(pkl_path+"vocab.pkl", "wb") as f:
#             pickle.dump([vocab_redux, word_counts], f, pickle.HIGHEST_PROTOCOL)
        
#     #negative sampler    
#     sampler = NegativeSampler(vocab, word_counts, warp=0.75)
#     rng = np.random.RandomState(random_seed)    
#     with open(inpath) as fi:
#         #peek at the first line to get the first user
#         curr_user, doc = fi.readline().replace("\"", "").replace("'","").split("\t")
#         #read file from the start
#         fi.seek(0,0)
#         user_docs = []
#         for line in fi:                        
#             user, doc = line.replace("\"", "").replace("'","").split("\t")            
#             #if we reach a new user, save the current one
#             if user!= curr_user:
#                 if len(user_docs) >= min_docs_user:
#                     save_user(curr_user, user_docs, sampler, rng, users_path, n_neg_samples)
#                 else:
#                     print("> IGNORED user: {}  ({})".format(user,len(user_docs)))
#                 #reset current user
#                 curr_user = user
#                 user_docs = []  
#             doc = doc.split(" ")            
#             doc_idx = [vocab[w] for w in doc if w in vocab]			    
#             if len(doc_idx) < MIN_DOC_LEN: continue
#             #accumulate all texts
#             user_docs.append(np.array(doc_idx, dtype=np.int32).reshape(1,-1))        
#         #save last user
#         if len(user_docs) >= min_docs_user:
#             save_user(curr_user, user_docs, sampler, rng, users_path, n_neg_samples)
#         else:
#             print("> IGNORED user: {}  ({})".format(user,len(user_docs)))

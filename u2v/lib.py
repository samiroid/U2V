# import lib_static
# import lib_context
# import encoders 

# import argparse
# # from ipdb import set_trace
# # import codecs
# from collections import Counter
# import math

import glob
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
import numpy as np
from numpy.random import RandomState
import time
from u2v.model import User2Vec

DEFAULT_WINDOW_SIZE=64
SHUFFLE_SEED=489
DEFAULT_BATCH_SIZE=128
MIN_DOC_LEN=2

# def build_data(inpath, outpath, encoder_type, embeddings_path=None, emb_encoding="latin-1", 
#                 min_word_freq=5, max_vocab_size=None, random_seed=SHUFFLE_SEED, 
#                 min_docs_user=2, reset=False):
    
#     outpath = f"{outpath}/{encoder_type}/"
#     pkl_path=outpath+"pkl/"
#     users_path=pkl_path+"users/"          
#     if reset:
#         shutil.rmtree(pkl_path, ignore_errors=True)
#         shutil.rmtree(users_path, ignore_errors=True)

#     if not os.path.exists(os.path.dirname(users_path)):
#         os.makedirs(os.path.dirname(users_path))   
    
#     if encoder_type == "w2v":
#         lib_static.build_data(inpath, outpath, embeddings_path, emb_encoding, 
#                 min_word_freq, max_vocab_size, random_seed, min_docs_user)
#     elif encoder_type == "bert":
#         lib_context.build_data(inpath, outpath, random_seed, min_docs_user)

def build_data(inpath, outpath, encoder, random_seed=SHUFFLE_SEED, 
                min_docs_user=1, reset=False):
    
    
    pkl_path=outpath+"pkl/"    
    if reset:
        shutil.rmtree(pkl_path, ignore_errors=True)    
    if not os.path.exists(os.path.dirname(pkl_path)):
        os.makedirs(os.path.dirname(pkl_path))   

    rng = np.random.RandomState(random_seed)    
    with open(inpath) as fi:
        #skip header
        next(fi)
        #peek at the first line to get the first user
        curr_user, doc = fi.readline().replace("\"", "").replace("\n","").replace("'","").split("\t")
        #read file from the start and skip header
        fi.seek(0,0)
        next(fi)
        user_docs = []
        doc_lens = []
        users = []
        #skip header        
        
        for line in fi:                        
            try:
                user, doc = line.replace("\"", "").replace("\n","").replace("'","").split("\t")            
            except ValueError:
                print(f"skipped line {line}")
                # from ipdb import set_trace; set_trace()
            #if we reach a new user, save the current one
            if user!= curr_user:
                if len(user_docs) >= min_docs_user:                    
                    save_user(curr_user, user_docs, doc_lens, rng, pkl_path)
                    users.append(curr_user)
                else:
                    print("> IGNORED user: {}  ({})".format(user,len(user_docs)))
                #reset current user
                curr_user = user
                user_docs = []  
                doc_lens = []
            
            if len(doc.split(" ")) < MIN_DOC_LEN: continue            
            doc_len, doc_idx = encoder.doc2idx(doc)            
            #accumulate all texts
            user_docs.append(doc_idx)        
            doc_lens.append(doc_len)
        #save last user
        if len(user_docs) >= min_docs_user:
            save_user(curr_user, user_docs, doc_lens,  rng, pkl_path)
            users.append(curr_user)
        else:
            print("> IGNORED user: {}  ({})".format(user,len(user_docs)))    
    print()
    with open(pkl_path+"users.txt","w") as fo:
        fo.write("\n".join(users))

def save_user(user_id, docs, doc_lens, rng, outpath):        
    sys.stdout.write("\r> saving user: {}  ({})".format(user_id,len(docs)))
    sys.stdout.flush()
    #shuffle the data
    shuf_idx = np.arange(len(docs))
    rng.shuffle(shuf_idx)
    docs_shuf = [docs[i] for i in shuf_idx]
    doc_lens_shuf = [doc_lens[i] for i in shuf_idx]
    
    with open(outpath+"idx_"+user_id, "wb") as fo:        
        pickle.dump([user_id, doc_lens_shuf, docs_shuf], fo)

def remix_samples(path, users):
    print("\n> remix negative samples")
    rng = RandomState(SHUFFLE_SEED)   
    fname_pos = path+"{}_pos.npy"
    fname_neg = path+"{}_neg.npy"
    for user in users:
        curr_user = np.load(fname_pos.format(user))
        other_users = users.copy()
        other_users.remove(user)
        #for each window 
        windows = []
        for x in curr_user.files:
            done=False
            curr_window_size = curr_user[x].shape[0]
            while not done:
                #sample a random user
                rand = rng.randint(len(other_users))
                rand_user = other_users[rand]
                rand_window = sample_user_window(fname_pos.format(rand_user), rng)
                if rand_window.shape[0] < curr_window_size:
                    # print("skip")
                    continue                
                elif rand_window.shape[0] > curr_window_size:
                    # print("trim")
                    rand_window = rand_window[:curr_window_size,:]                
                windows.append(rand_window)
                done=True
        # print(len(curr_user.files))
        # from ipdb import set_trace; set_trace()
        with open(fname_neg.format(user), "wb") as fi:
            np.savez(fi, *windows)        
    
def mean_word_embeddings(path, users):
    print("> mean word embeddings")
    rng = RandomState(SHUFFLE_SEED)   
    fname_pos = path+"/pkl/{}_pos.npy"
    fname_mean = path+"/txt/mean-{}.txt"
    for user in users:
        curr_user = np.load(fname_pos.format(user))                
        #sum of all embeddings
        emb_dim = curr_user[curr_user.files[0]].shape[1]
        sum_emb = np.zeros(emb_dim)
        total_size = 0
        for x in curr_user.files:            
            window_size = curr_user[x].shape[0]
            total_size+=window_size
            window = curr_user[x]
            sum_emb+=np.sum(window, axis=0)
                
        mean_emb = sum_emb/total_size            
            
        with open(fname_mean.format(user),"w") as fo:              
            fo.write('%d %d\n' % (1, emb_dim))            
            e = ' '.join(map(lambda x: str(x), mean_emb))
            fo.write('%s %s\n' % (user, e))
        
    stich_embeddings(path+"/txt/", path, emb_dim, "mean")


def sample_user_window(path, rng):
    x = np.load(path)
    windows = x.files
    rng.shuffle(windows)
    rand_window = windows[0]
    # print(rand_window)
    return x[rand_window]

def encode_users(path, encoder, window_size=DEFAULT_WINDOW_SIZE):
    # path = f"{path}/{encoder_type}/"
    pkl_path= f"{path}/pkl/"
    txt_path= f"{path}/txt/"
    if not os.path.exists(os.path.dirname(pkl_path)):
        os.makedirs(os.path.dirname(pkl_path)) 
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path)) 
    users = encoder.encode(path, pkl_path, window_size)    
    remix_samples(pkl_path, users)
    mean_word_embeddings(path, users)
    

def stich_embeddings(inpath, outpath, emb_dim, run_id=""):
    if run_id:
        user_embeddings = list(glob.glob(f"{inpath}/{run_id}-*"))
        outpath = f"{outpath}U_{run_id}.txt"        
    else:        
        user_embeddings = list(glob.glob(f"{inpath}/*"))
        outpath = f"{outpath}U.txt"

    print("[writing embeddings to {}]".format(outpath))
    with open(outpath,"w") as fo:            
        fo.write("{}\t{}\n".format(len(user_embeddings), emb_dim))
        for u in user_embeddings:
            with open(u, "r") as fi:
                l = fi.readlines()[1]
            fo.write(l)

def train_model(path, run_id=None, batch_size=100, epochs=20, initial_lr=0.001, margin=1, validation_split=0.8, reset=False, device=None):  
    print("\ntraining...")
    st = time.time()
    # path = f"{path}/{encoder_type}/"
    txt_path = path+"/txt/"
    if reset:
        shutil.rmtree(txt_path, ignore_errors=True)    
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path))       
    with open(path+"/pkl/users.txt") as fi:
        users = [u.replace("\n","") for u in fi.readlines()]
    
    random.shuffle(users)
    cache = set([os.path.basename(f).replace(".txt","") for f in Path(txt_path).iterdir()])
    # from ipdb import set_trace; set_trace()
    emb_dim = None    
    val_perfs = []
    for user in users:    
        #account for cases where a run_id is used to prefix the file
        user_check = run_id+"-"+user if run_id else user
        if user_check in cache:
            print("cached embedding: {}".format(user))
            continue
        user_fname = "{}/pkl/{}_{}.npy" 
        f_pos = open(user_fname.format(path, user, "pos"), "rb")
        pos_samples = np.load(f_pos, allow_pickle=True)            
        f_neg = open(user_fname.format(path, user, "neg"), "rb")
        neg_samples = np.load(f_neg, allow_pickle=True)
        
        emb_dim = pos_samples[pos_samples.files[0]].shape[1] 
        f = User2Vec(user, emb_dim, txt_path, margin=margin, initial_lr=initial_lr, 
                    epochs=epochs, device=device, batch_size=batch_size, validation_split=validation_split, run_id=run_id)   
        val_perf = f.fit(pos_samples, neg_samples)
        val_perfs.append(val_perf)        
        f_pos.close()
        f_neg.close()
        
    ft = time.time()
    et = ft - st
    print(f"total time: {round(et,3)}")
    max_val = 0
    min_val = 0
    mean_val = 0
    std_val = 0
    if len(val_perfs) > 0:
        max_val = round(max(val_perfs),3)
        min_val = round(min(val_perfs),3)
        mean_val = round(np.mean(val_perfs),3)
        std_val = round(np.std(val_perfs),3)
    print(f"avg val: {mean_val} ({std_val}) [{max_val};{min_val}]")
    if emb_dim:        
        stich_embeddings(txt_path, path, emb_dim, run_id)

    
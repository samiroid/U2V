import lib_static
import lib_context

# import argparse
# # from ipdb import set_trace
# import glob
import os
import pickle
import random
import shutil
import sys
# # import codecs
# from collections import Counter
from pathlib import Path
# import math
import numpy as np
from numpy.random import RandomState
import time
from model import User2Vec

DEFAULT_BLOCK_SIZE=64
SHUFFLE_SEED=489


def build_data(inpath, outpath, encoder_type, embeddings_path=None, emb_encoding="latin-1", 
                min_word_freq=5, max_vocab_size=None, random_seed=SHUFFLE_SEED, 
                min_docs_user=2, reset=False):
    
    pkl_path=outpath+"pkl/"
    users_path=pkl_path+"users/"          
    if reset:
        shutil.rmtree(pkl_path, ignore_errors=True)
        shutil.rmtree(users_path, ignore_errors=True)

    if not os.path.exists(os.path.dirname(users_path)):
        os.makedirs(os.path.dirname(users_path))   
    
    if encoder_type == "w2v":
        lib_static.build_data(inpath, outpath, embeddings_path, emb_encoding, 
                min_word_freq, max_vocab_size, random_seed, min_docs_user)
    elif encoder_type == "bert":
        lib_context.build_data(inpath, outpath, random_seed, min_docs_user)
    
def save_user(user_id, docs, doc_lens, rng, outpath, split=0.8):        
    sys.stdout.write("\r> saving user: {}  ({})".format(user_id,len(docs)))
    sys.stdout.flush()
    #shuffle the data
    shuf_idx = np.arange(len(docs))
    rng.shuffle(shuf_idx)
    docs_shuf = [docs[i] for i in shuf_idx]
    split_idx = int(len(docs)*split)
    docs_train = docs_shuf[:split_idx]
    docs_val = docs_shuf[split_idx:]
    
    with open(outpath+"idx_"+user_id, "wb") as fo:        
        pickle.dump([user_id, doc_lens, docs_train, docs_val], fo)

def remix_samples(inpath, users):
    print("\n> remix negative samples")
    rng = RandomState(SHUFFLE_SEED)   
    fname_pos = inpath+"{}_pos.npy"
    fname_neg = inpath+"{}_neg.npy"
    for user in users:
        curr_user = np.load(fname_pos.format(user))
        other_users = users.copy()
        other_users.remove(user)
        #for each block 
        blocks = []
        for x in curr_user.files:
            done=False
            curr_block_size = curr_user[x].shape[0]
            while not done:
                #sample a random user
                rand = rng.randint(len(other_users))
                rand_user = other_users[rand]
                rand_block = sample_user_block(fname_pos.format(rand_user), rng)
                if rand_block.shape[0] < curr_block_size:
                    # print("skip")
                    continue                
                elif rand_block.shape[0] > curr_block_size:
                    # print("trim")
                    rand_block = rand_block[:curr_block_size,:]                
                blocks.append(rand_block)
                done=True
        # print(len(curr_user.files))
        # from ipdb import set_trace; set_trace()
        with open(fname_neg.format(user), "wb") as fi:
            np.savez(fi, *blocks)        
    
    print()

def sample_user_block(path, rng):
    x = np.load(path)
    blocks = x.files
    rng.shuffle(blocks)
    rand_block = blocks[0]
    # print(rand_block)
    return x[rand_block]

def encode_users(inpath, encoder_type, block_size=DEFAULT_BLOCK_SIZE):

    outpath= f"{inpath}/pkl/users/{encoder_type}/"
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath)) 

    if encoder_type == "w2v":
        users = lib_static.word2vec_features(inpath, outpath, block_size=block_size)
    elif encoder_type == "bert":
        users = lib_context.BERT_features(inpath, outpath, block_size=block_size)
    else:
        raise NotImplementedError
    remix_samples(outpath, users)



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
    print("\ntraining...")
    st = time.time()
    txt_path = f"{path}/txt/{encoder}/"
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
        user_fname = "{}/pkl/users/{}/{}_{}.npy" 

        f_pos = open(user_fname.format(path, encoder, user, "pos"), "rb")
        pos_samples = np.load(f_pos, allow_pickle=True)            
        f_neg = open(user_fname.format(path, encoder, user, "neg"), "rb")
        neg_samples = np.load(f_neg, allow_pickle=True)
        f_val = open(user_fname.format(path, encoder, user, "val"), "rb")
        val_samples = np.load(f_val, allow_pickle=True)        
        emb_dim = val_samples.shape[1]        
        f = User2Vec(user, emb_dim, txt_path, margin=margin, initial_lr=initial_lr, epochs=epochs, device=device, batch_size=100)   
        f.fit(pos_samples, neg_samples, val_samples)
        # break
        f_pos.close()
        f_neg.close()
        f_val.close()
    ft = time.time()
    et = ft - st
    print(f"total time: {round(et,3)}")
    if emb_dim:
        stich_embeddings(txt_path, path+"{}_U.txt".format(encoder), emb_dim)

    
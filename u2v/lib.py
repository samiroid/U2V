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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DEFAULT_WINDOW_SIZE=64
SHUFFLE_SEED=489
DEFAULT_BATCH_SIZE=128
MIN_DOC_LEN=2

def build_data(inpath, outpath, encoder, random_seed=SHUFFLE_SEED, 
                min_docs_user=1, max_docs_user=None, reset=False):
    
    
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
            #if we reach a new user, save the current one
            if user!= curr_user:
                if len(user_docs) >= min_docs_user:                    
                    save_user(curr_user, user_docs, doc_lens, rng, pkl_path, max_docs_user)
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
            save_user(curr_user, user_docs, doc_lens, rng, pkl_path, max_docs_user)
            users.append(curr_user)
        else:
            print("> IGNORED user: {}  ({})".format(user,len(user_docs)))    
    print()
    with open(pkl_path+"users.txt","w") as fo:
        fo.write("\n".join(users))

def save_user(user_id, docs, doc_lens, rng, outpath, max_docs=None):        
    if max_docs:
        docs = docs[:max_docs]
        doc_lens = doc_lens[:max_docs]
    sys.stdout.write("\r> saving user: {}  ({})".format(user_id,len(docs)))
    sys.stdout.flush()
    #shuffle the data
    shuf_idx = np.arange(len(docs))
    rng.shuffle(shuf_idx)
    docs_shuf = [docs[i] for i in shuf_idx]
    doc_lens_shuf = [doc_lens[i] for i in shuf_idx]
    
    with open(outpath+"idx_"+user_id, "wb") as fo:        
        pickle.dump([user_id, doc_lens_shuf, docs_shuf], fo)

def remix_samples(path, users, encoder_name, cache=False):
    print("\n> remix negative samples")
    rng = RandomState(SHUFFLE_SEED)   
    fname_pos = path+"{}_{}_pos.npy"
    fname_neg = path+"{}_{}_neg.npy"

    with tqdm(users, unit="users") as pbar:
        for user in pbar:
        # for user in users:
            if cache and os.path.isfile(fname_neg.format(encoder_name, user)):
                pbar.set_description(f"user {user} in cache")
                continue    
            # else:
            #     pbar.set_description(f"sampling user {user}")
            curr_user = np.load(fname_pos.format(encoder_name, user))
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
                    rand_window = sample_user_window(fname_pos.format(encoder_name, rand_user), rng)
                    if rand_window.shape[0] < curr_window_size:                    
                        continue                
                    elif rand_window.shape[0] > curr_window_size:
                        #trim if needed
                        rand_window = rand_window[:curr_window_size,:]                
                    windows.append(rand_window)
                    done=True        
            with open(fname_neg.format(encoder_name, user), "wb") as fi:
                np.savez(fi, *windows)        
    
def mean_word_embeddings(path, users, encoder_name):
    print("> mean word embeddings")
    rng = RandomState(SHUFFLE_SEED)   
    fname_pos = path+"/pkl/{}_{}_pos.npy"
    fname_mean = path+"/txt/{}_mean_{}.txt"
    for user in users:
        curr_user = np.load(fname_pos.format(encoder_name, user))                
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
            
        with open(fname_mean.format(encoder_name, user),"w") as fo:              
            fo.write('%d %d\n' % (1, emb_dim))            
            e = ' '.join(map(lambda x: str(x), mean_emb))
            fo.write('%s %s\n' % (user, e))
        
    stich_embeddings(path+"/txt/", encoder_name, path, emb_dim, "mean")


def sample_user_window(path, rng):
    x = np.load(path)
    windows = x.files
    rng.shuffle(windows)
    rand_window = windows[0]
    # print(rand_window)
    return x[rand_window]

def encode_users(path, encoder, window_size=DEFAULT_WINDOW_SIZE, cache=True):
    # path = f"{path}/{encoder_type}/"
    pkl_path= f"{path}/pkl/"
    txt_path= f"{path}/txt/"
    if not os.path.exists(os.path.dirname(pkl_path)):
        os.makedirs(os.path.dirname(pkl_path)) 
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path)) 
    users = encoder.encode(path, pkl_path, window_size, cache)    
    remix_samples(pkl_path, users, encoder.encoder_name, cache)
    mean_word_embeddings(path, users, encoder.encoder_name)
    

def stich_embeddings(inpath, encoder_name, outpath, emb_dim, run_id):
    if run_id:
        user_embeddings = list(glob.glob(f"{inpath}/{encoder_name}_{run_id}_*"))
        outpath = f"{outpath}U_{encoder_name}_{run_id}.txt"            

    print("[writing embeddings to {}]".format(outpath))
    with open(outpath,"w") as fo:            
        fo.write("{}\t{}\n".format(len(user_embeddings), emb_dim))
        for u in user_embeddings:
            with open(u, "r") as fi:
                l = fi.readlines()[1]
            fo.write(l)

def train_model(path, encoder, run_id, logs_path=None, batch_size=100, epochs=20, initial_lr=0.001, margin=1, validation_split=0.8, cache=False, device=None):  
    print("\ntraining...")
    st = time.time()
    txt_path = path+"/txt/"    
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path))       
    with open(path+"/pkl/users.txt") as fi:
        users = [u.replace("\n","") for u in fi.readlines()]
    random.shuffle(users)
    tmp_users = glob.glob(f"{txt_path}/{encoder.encoder_name}_{run_id}*")
    cached_users = []
    if not cache:
        #remove all existing files
        for f in tmp_users: 
            os.remove(f)        
    else:    
        cached_users = set([os.path.basename(f).replace(".txt","") for f in  tmp_users])    
    emb_dim = None    
    val_perfs = []
    tensorboard = SummaryWriter(logs_path)
    for user in users:        
        user_check = f"{encoder.encoder_name}_{run_id}_{user}"  
        if user_check in cached_users:
            print("cached embedding: {}".format(user))
            continue
        user_fname = "{}/pkl/{}_{}_{}.npy" 
        f_pos = open(user_fname.format(path, encoder.encoder_name, user, "pos"), "rb")
        pos_samples = np.load(f_pos, allow_pickle=True)            
        f_neg = open(user_fname.format(path, encoder.encoder_name, user, "neg"), "rb")
        neg_samples = np.load(f_neg, allow_pickle=True)
        
        emb_dim = pos_samples[pos_samples.files[0]].shape[1] 
        f = User2Vec(user, encoder.encoder_name,run_id, emb_dim, txt_path,
                    logs_path=logs_path, margin=margin, initial_lr=initial_lr, epochs=epochs, device=device, batch_size=batch_size, validation_split=validation_split)   
        val_perf = f.fit(pos_samples, neg_samples)
        val_perfs.append(val_perf)        
        f_pos.close()
        f_neg.close()
        # tensorboard.add_scalars("val prob",{user:val_perf})    
    ft = time.time()
    et = ft - st
    print(f"total time: {round(et,3)}")
    max_val = 0
    min_val = 0
    mean_val = 0
    std_val = 0
    tensorboard.add_histogram("val prob",np.array(val_perfs))
    if len(val_perfs) > 0:
        max_val = round(max(val_perfs),3)
        min_val = round(min(val_perfs),3)
        mean_val = round(np.mean(val_perfs),3)
        std_val = round(np.std(val_perfs),3)
    print(f"avg val: {mean_val} ({std_val}) [{max_val};{min_val}]")
    if emb_dim:        
        stich_embeddings(txt_path, encoder.encoder_name, path, emb_dim, run_id)

    
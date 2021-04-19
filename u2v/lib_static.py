import glob
import os
import pickle
import shutil
import sys
# import codecs
from collections import Counter
import math
import numpy as np

from tadat.core import embeddings

MIN_DOC_LEN=2
MIN_DOCS = 2
DEFAULT_BLOCK_SIZE=64
SHUFFLE_SEED=489

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
                min_word_freq=5, max_vocab_size=None, random_seed=123, 
                min_docs_user=2):
    pkl_path=outpath+"pkl/"
    users_path=pkl_path+"users/"      
    
    # if reset:
    #     shutil.rmtree(pkl_path, ignore_errors=True)
    #     shutil.rmtree(users_path, ignore_errors=True)

    # if not os.path.exists(os.path.dirname(users_path)):
    #     os.makedirs(os.path.dirname(users_path))   

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
        curr_user, doc = fi.readline().replace("\"", "").replace("\n","").replace("'","").split("\t")
        #read file from the start
        fi.seek(0,0)
        user_docs = []
        doc_lens = []
        users = []
        for line in fi:                        
            user, doc = line.replace("\"", "").replace("\n","").replace("'","").split("\t")            
            #if we reach a new user, save the current one
            if user!= curr_user:
                if len(user_docs) >= min_docs_user:
                    # max_doc_len = max(doc_lens)
                    save_user(curr_user, user_docs, doc_lens, rng, users_path)
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
            # max_doc_len = max(doc_lens)
            save_user(curr_user, user_docs, doc_lens,  rng, users_path)
            users.append(curr_user)
        else:
            print("> IGNORED user: {}  ({})".format(user,len(user_docs)))    
    print()
    with open(pkl_path+"users.txt","w") as fo:
        fo.write("\n".join(users))
    
    # 
    
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


def word2vec_features(inpath, outpath, block_size=DEFAULT_BLOCK_SIZE):
    """
        save features in blocks of block_size 
    """
    
    inpath=inpath+"pkl/"
    print("\n> word2vec features")    
    with open(inpath+"word_emb.npy", "rb") as f:
        E = np.load(f)        
    users = []
    for user_path in glob.glob(inpath+"/users/idx_*"):
        with open(user_path, "rb") as fi:
            user_id, doc_lens, train, validation = pickle.load(fi) 
        users.append(user_id)
        #concatenate all documents into a  single vector
        train = np.concatenate(train, axis=None)        
        validation = np.concatenate(validation, axis=None)
        #validation data goes into a single block
        X_validation = E[:, validation].T 
        with open(outpath+f"/{user_id}_val.npy", "wb") as fi:
            np.save(fi, X_validation)
        n_blocks = math.ceil(len(train)/block_size)                
        blocks = []
        for i in range(n_blocks):
            block = train[i*block_size:(i+1)*block_size]                
            E_block = E[:, block].T
            blocks.append(E_block)
        # from ipdb import set_trace; set_trace()
        with open(outpath+f"/{user_id}_pos.npy", "wb") as fi:
            np.savez(fi, *blocks)        
        sys.stdout.write("\r> features | user: {}".format(user_id))
        sys.stdout.flush()
    return users    


    
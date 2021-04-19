import glob
import os
import pickle
import random
import shutil
import sys
# import codecs
import numpy as np

from transformers import AutoTokenizer, AutoModel
DEFAULT_BERT_MODEL = 'bert-base-uncased'
BERT_MAX_INPUT=512
DEFAULT_BATCH_SIZE = 64

MIN_DOC_LEN=2
MIN_DOCS = 2
DEFAULT_BLOCK_SIZE=64
SHUFFLE_SEED=489

def build_data(inpath, outpath, random_seed=SHUFFLE_SEED, min_docs_user=2, pretrained_model=DEFAULT_BERT_MODEL):
    
    pkl_path=outpath+"pkl/"
    users_path=pkl_path+"users/"  
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # from pdb import set_trace; set_trace()

    
    
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
            
            if len(doc.split(" ")) < MIN_DOC_LEN: continue
            
            doc_len, doc_idx = bertdoc2idx(tokenizer, doc)            
            #accumulate all texts
            user_docs.append(doc_idx)        
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


def bertdoc2idx(tokenizer, doc):
    bertify = "[CLS] {} [SEP]"  
    tokenized_doc = tokenizer.tokenize(bertify.format(doc))
    idxs = tokenizer.convert_tokens_to_ids(tokenized_doc)       
    doc_len = len(idxs)
    if doc_len > BERT_MAX_INPUT: idxs = idxs[:BERT_MAX_INPUT]
    pad_size = BERT_MAX_INPUT - doc_len
    #add padding to indexed tokens
    idxs+=[0] * pad_size         
    # print(tokenizer.max_model_input_sizes)    
    return doc_len, idxs

def sequence_features(docs, tokenizer, encoder, max_input_size, device):    
    docs_tensor = []    
    tokenized_docs = []    
    bertify = "[CLS] {} [SEP]"  
    tokenized_docs = [tokenizer.tokenize(bertify.format(doc)) for doc in docs]     
    #count the document lengths  
    max_len = max([len(d) for d in tokenized_docs]) 
    #document cannot exceed BERT input matrix size 
    max_len = min(max_input_size, max_len)
    print("[max len: {}]".format(max_len))
    for doc in tokenized_docs:   
        # Convert tokens to vocabulary indices
        idxs = tokenizer.convert_tokens_to_ids(doc)        
        
        #truncate sentences longer than what BERT supports
        if len(idxs) > max_len: idxs = idxs[:max_len]
        pad_size = max_len - len(idxs)
        #add padding to indexed tokens
        idxs+=[0] * pad_size        
        docs_tensor.append(torch.tensor([idxs]))        
    
    # Convert inputs to PyTorch tensors
    docs_tensor = torch.cat(docs_tensor)
    segments_tensor = torch.zeros_like(docs_tensor)            
    #set encoder to eval mode
    encoder.eval()
    #device    
    docs_tensor = docs_tensor.to(device)
    segments_tensor =  segments_tensor.to(device)
    encoder = encoder.to(device)
    with torch.no_grad():        
        pool_features, cls_features = encoder(docs_tensor, 
                                            token_type_ids=segments_tensor)    
        pool_features = pool_features.sum(axis=1)
    return cls_features.cpu().numpy(), pool_features.cpu().numpy()

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
    
    with open(outpath+"sub_"+user_id, "wb") as fo:        
        pickle.dump([user_id, doc_lens, docs_train, docs_val], fo)

def BERT_features(inpath, outpath, pretrained_model=DEFAULT_BERT_MODEL,
                device="cpu", block_size=DEFAULT_BLOCK_SIZE):
    """
        save features in blocks of block_size 
    """
    inpath=inpath+"pkl/"
    print("\n> BERT features")        
    
    pkl_path=outpath+"pkl/"
    users_path=pkl_path+"users/"  
    
    encoder = AutoModel.from_pretrained(pretrained_model).to(device)
    #set encoder to eval mode
    encoder.eval()

    users = []
    for user_path in glob.glob(inpath+"/users/sub_*"):
        with open(user_path, "rb") as fi:
            user_id, doc_lens, train, validation = pickle.load(fi) 
        users.append(user_id)
        docs_tensor = torch.cat([torch.tensor([t]) for t in train[:10]])
        segments_tensor = torch.zeros_like(docs_tensor)            
        
        #device    
        docs_tensor = docs_tensor.to(device)
        segments_tensor =  segments_tensor.to(device)
        
        with torch.no_grad():      
            print("encoding now")  
            pool_features, cls_features = encoder(docs_tensor, 
                                                token_type_ids=segments_tensor)    
            pool_features = pool_features.cpu().numpy()

        print(pool_features)
        return



        # sys.stdout.write("\r> features | user: {}".format(user_id))
        # sys.stdout.flush()
        
    return users 
        
        
        
        # #concatenate all documents into a  single vector
        # train = np.concatenate(train, axis=None)        
        # validation = np.concatenate(validation, axis=None)
        # #validation data goes into a single block
        # X_validation = E[:, validation].T 
        # with open(outpath+f"/{user_id}_val.npy", "wb") as fi:
        #     np.save(fi, X_validation)
        # n_blocks = math.ceil(len(train)/block_size)                
        # blocks = []
        # for i in range(n_blocks):
        #     block = train[i*block_size:(i+1)*block_size]                
        #     E_block = E[:, block].T
        #     blocks.append(E_block)
        # # from ipdb import set_trace; set_trace()
        # with open(outpath+f"/{user_id}_pos.npy", "wb") as fi:
        #     np.savez(fi, *blocks)        
    
    





import torch.nn as nn
import sys
import random 
import os
import uuid 
import torch 



####################################################################

def get_encoder(pretrained_model=DEFAULT_BERT_MODEL, hidden_states=False):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    # Load pre-trained model (weights)
    model = AutoModel.from_pretrained(pretrained_model, output_hidden_states=hidden_states)
    return tokenizer, model

def encode_sequences(docs, tokenizer=None, encoder=None, max_input_size=BERT_MAX_INPUT,
                    cls_features=False, batchsize=DEFAULT_BATCH_SIZE, device="cpu"):  
    #BERT
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BERT_MODEL)
    if not encoder:
        encoder = AutoModel.from_pretrained(DEFAULT_BERT_MODEL, output_hidden_states=False)
    
    feature_vectors = []    
    n_batches = int(len(docs)/batchsize)+1
    for j in range(n_batches):
        batch = docs[batchsize*j:batchsize*(j+1)]
        if len(batch) > 0:
            sys.stdout.write("\nbatch:{}/{} (size: {})".format(j+1,n_batches, str(len(batch))))
            sys.stdout.flush()
            cls_vec, pool_vec = sequence_features(batch,tokenizer, encoder, max_input_size, device)            
            feats = cls_vec if cls_features else pool_vec                       
            feature_vectors.append(feats)                
    feature_vectors = np.vstack(feature_vectors)

    return feature_vectors

def sequence_features(docs, tokenizer, encoder, max_input_size, device):    
    docs_tensor = []    
    tokenized_docs = []    
    bertify = "[CLS] {} [SEP]"  
    tokenized_docs = [tokenizer.tokenize(bertify.format(doc)) for doc in docs]     
    #count the document lengths  
    max_len = max([len(d) for d in tokenized_docs]) 
    #document cannot exceed BERT input matrix size 
    max_len = min(max_input_size, max_len)
    print("[max len: {}]".format(max_len))
    for doc in tokenized_docs:   
        # Convert tokens to vocabulary indices
        idxs = tokenizer.convert_tokens_to_ids(doc)        
        #truncate sentences longer than what BERT supports
        if len(idxs) > max_len: idxs = idxs[:max_len]
        pad_size = max_len - len(idxs)
        #add padding to indexed tokens
        idxs+=[0] * pad_size        
        docs_tensor.append(torch.tensor([idxs]))        
    
    # Convert inputs to PyTorch tensors
    docs_tensor = torch.cat(docs_tensor)
    segments_tensor = torch.zeros_like(docs_tensor)            
    #set encoder to eval mode
    encoder.eval()
    #device    
    docs_tensor = docs_tensor.to(device)
    segments_tensor =  segments_tensor.to(device)
    encoder = encoder.to(device)
    with torch.no_grad():        
        pool_features, cls_features = encoder(docs_tensor, 
                                            token_type_ids=segments_tensor)    
        pool_features = pool_features.sum(axis=1)
    return cls_features.cpu().numpy(), pool_features.cpu().numpy()
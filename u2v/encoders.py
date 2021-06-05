from abc import ABC, abstractmethod
from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import torch
from torch._C import set_num_threads
from transformers import AutoTokenizer, AutoModel
import math
import pickle
import sys
import glob
from collections import Counter
import fasttext
from tadat.core import embeddings
import os
from tqdm import tqdm
BERT_MAX_INPUT = 512
ELMO_MAX_INPUT = 128
class Encoder(ABC):

    def doc2idx(self, doc):
        doc_splt = doc.split(" ")
        doc_len = len(doc_splt)
        return doc_len, doc_splt
    
    @abstractmethod
    def encode(self, inpath, outpath, window_size):
        pass

    

class BERTEncoder(Encoder):    

    def __init__(self, pretrained_weights, encoder_batchsize, device) -> None:
        super().__init__()
        weights = {
        "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
        "base": "bert-base-uncased"
        }
        self.pretrained_weights = weights.get(pretrained_weights, pretrained_weights)
        self.encoder_batchsize = encoder_batchsize
        self.device = device        
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_weights)
        self.encoder = AutoModel.from_pretrained(self.pretrained_weights).to(self.device)
        #set encoder to eval mode
        self.encoder.eval()
        self.encoder_name = f"bert_{pretrained_weights}"

    def doc2idx(self, doc):
        bertify = "[CLS] {} [SEP]"  
        tokenized_doc = self.tokenizer.tokenize(bertify.format(doc))
        idxs = self.tokenizer.convert_tokens_to_ids(tokenized_doc)       
        doc_len = len(idxs)
        if doc_len > BERT_MAX_INPUT: 
            idxs = idxs[:BERT_MAX_INPUT]
            doc_len = BERT_MAX_INPUT

        pad_size = BERT_MAX_INPUT - doc_len
        #add padding to indexed tokens
        idxs+=[0] * pad_size         
        # print(tokenizer.max_model_input_sizes)    
        return doc_len, idxs
    
    def encode(self, inpath, outpath, window_size):
        inpath=inpath+"pkl/"        
        
        print("\n> BERT features ({})".format(self.pretrained_weights))        
        
        users = []
        # for user_path in glob.glob(inpath+"/idx_*"):
        data = glob.glob(inpath+"/idx_*")
        
        with tqdm(data, unit="users") as pbar:
            for user_path in pbar:
                with open(user_path, "rb") as fi:
                    user_id, doc_lens, docs = pickle.load(fi) 
                users.append(user_id)        
                encoded_tensors = []
                n_batches = math.ceil(len(docs)/self.encoder_batchsize)
                for j in range(n_batches):
                    batch_docs = docs[self.encoder_batchsize*j:self.encoder_batchsize*(j+1)]
                    batch_lens = doc_lens[self.encoder_batchsize*j:self.encoder_batchsize*(j+1)]
                    if len(batch_docs) > 0:
                        # sys.stdout.write("\rbatch:{}/{} (size: {})".format(j+1,n_batches, str(len(batch_docs))))
                        # sys.stdout.flush()
                        pbar.set_description(f"user: {user_id} | batch:{j+1}/{n_batches} ({len(batch_docs)})")
                        #encode it
                        docs_tensor = torch.cat([torch.tensor([t]) for t in batch_docs])
                        segments_tensor = torch.zeros_like(docs_tensor)                           
                        docs_tensor = docs_tensor.to(self.device)
                        segments_tensor = segments_tensor.to(self.device)                    
                        with torch.no_grad():              
                            model_out = self.encoder(docs_tensor, token_type_ids=segments_tensor)    
                            Z, cls = model_out.to_tuple()
                            # from pdb import set_trace; set_trace()
                            Z = Z.cpu().numpy()
                        #append encoded docs
                        for l, z in zip(batch_lens,Z):
                            #get rid of [CLS] [SEP] and padding dimensions
                            z_trunc = z[1:(l-1)]
                            encoded_tensors.append(z_trunc)
                #convert all doc tensors into a single mega tensor
                mega_tensor = np.concatenate(encoded_tensors, axis=0)        
                #slice the mega tensor into slices of window_size
                n_windows = math.ceil(mega_tensor.shape[0]/window_size)
                # from pdb import set_trace; set_trace()
                windows = []
                for j in range(n_windows):
                    batch_docs = mega_tensor[window_size*j:window_size*(j+1) , :]
                    windows.append(batch_docs)
                
                # sys.stdout.write("\r> features | user: {}".format(user_id))
                # sys.stdout.flush()
                # from pdb import set_trace; set_trace()
                with open(outpath+f"/{self.encoder_name}_{user_id}_pos.npy", "wb") as fi:
                    np.savez(fi, *windows)        
            # print(docs_tensor)        
        return users

    # def encode(self, inpath, outpath, window_size):
    #     inpath=inpath+"pkl/"        
        
    #     print("\n> BERT features ({})".format(self.pretrained_weights))        
        
    #     users = []
    #     for user_path in glob.glob(inpath+"/idx_*"):
    #         with open(user_path, "rb") as fi:
    #             user_id, doc_lens, docs = pickle.load(fi) 
    #         users.append(user_id)        
    #         encoded_tensors = []
    #         n_batches = math.ceil(len(docs)/self.encoder_batchsize)
    #         for j in range(n_batches):
    #             batch_docs = docs[self.encoder_batchsize*j:self.encoder_batchsize*(j+1)]
    #             batch_lens = doc_lens[self.encoder_batchsize*j:self.encoder_batchsize*(j+1)]
    #             if len(batch_docs) > 0:
    #                 sys.stdout.write("\rbatch:{}/{} (size: {})".format(j+1,n_batches, str(len(batch_docs))))
    #                 sys.stdout.flush()
    #                 #encode it
    #                 docs_tensor = torch.cat([torch.tensor([t]) for t in batch_docs])
    #                 segments_tensor = torch.zeros_like(docs_tensor)                           
    #                 docs_tensor = docs_tensor.to(self.device)
    #                 segments_tensor = segments_tensor.to(self.device)                    
    #                 with torch.no_grad():              
    #                     model_out = self.encoder(docs_tensor, token_type_ids=segments_tensor)    
    #                     Z, cls = model_out.to_tuple()
    #                     # from pdb import set_trace; set_trace()
    #                     Z = Z.cpu().numpy()
    #                 #append encoded docs
    #                 for l, z in zip(batch_lens,Z):
    #                     #get rid of [CLS] [SEP] and padding dimensions
    #                     z_trunc = z[1:(l-1)]
    #                     encoded_tensors.append(z_trunc)
    #         #convert all doc tensors into a single mega tensor
    #         mega_tensor = np.concatenate(encoded_tensors, axis=0)        
    #         #slice the mega tensor into slices of window_size
    #         n_windows = math.ceil(mega_tensor.shape[0]/window_size)
    #         # from pdb import set_trace; set_trace()
    #         windows = []
    #         for j in range(n_windows):
    #             batch_docs = mega_tensor[window_size*j:window_size*(j+1) , :]
    #             windows.append(batch_docs)
            
    #         sys.stdout.write("\r> features | user: {}".format(user_id))
    #         sys.stdout.flush()
    #         # from pdb import set_trace; set_trace()
    #         with open(outpath+f"/{self.encoder_name}_{user_id}_pos.npy", "wb") as fi:
    #             np.savez(fi, *windows)        
    #         # print(docs_tensor)        
    #     return users

class ELMoEncoder(Encoder):

    def __init__(self, encoder_batchsize, device, pretrained_weights="small") -> None:
        super().__init__()
        pw = {
                "small": {
                    "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",

                    "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
                },

                "medium": {
                    "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json",

                    "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
                },

                "original": {
                    "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",

                    "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
                }
        }
        # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

        options_file = pw[pretrained_weights]["options"] 
        weight_file = pw[pretrained_weights]["weights"] 

        elmo = Elmo(options_file=options_file, weight_file=weight_file, 
                    num_output_representations=1, dropout=0)
        self.encoder = elmo.to(device)
        self.encoder_batchsize = encoder_batchsize
        self.device = device
        self.encoder_name = "elmo_"+pretrained_weights

    def encode(self, inpath, outpath, window_size):
        inpath=inpath+"pkl/"        
        
        print("\n> ELMo features")        
        
        users = []
        data = glob.glob(inpath+"/idx_*")
        
        with tqdm(data, unit="users") as pbar:
            for user_path in pbar:
                with open(user_path, "rb") as fi:
                    user_id, doc_lens, docs = pickle.load(fi) 
                users.append(user_id)        
                encoded_docs = []
                n_batches = math.ceil(len(docs)/self.encoder_batchsize)
                for j in range(n_batches):
                    batch_docs = docs[self.encoder_batchsize*j:self.encoder_batchsize*(j+1)]
                    batch_lens = doc_lens[self.encoder_batchsize*j:self.encoder_batchsize*(j+1)]
                    if len(batch_docs) > 0:
                        char_idxs = batch_to_ids(batch_docs).to(self.device)         
                        char_idxs = char_idxs[:,:ELMO_MAX_INPUT,:]
                        pbar.set_description(f"user: {user_id} | batch:{j+1}/{n_batches} ({len(batch_docs)})")
                        #encode it                    
                        with torch.no_grad():   
                            output = self.encoder(char_idxs)
                        Z = output["elmo_representations"][0]
                        Z = Z.cpu().numpy()
                        #append encoded docs
                        for l, z in zip(batch_lens,Z):
                            #get rid of padding dimensions
                            z_trunc = z[:l]
                            encoded_docs.append(z_trunc)
                            # from ipdb import set_trace; set_trace()
                #convert all doc tensors into a single mega tensor
                mega_tensor = np.concatenate(encoded_docs, axis=0)        
                #slice the mega tensor into slices of window_size
                n_windows = math.ceil(mega_tensor.shape[0]/window_size)
                # from pdb import set_trace; set_trace()
                windows = []
                for j in range(n_windows):
                    batch_docs = mega_tensor[window_size*j:window_size*(j+1) , :]
                    windows.append(batch_docs)
                
                # print("\r> features | user: {}".format(user_id))
                # sys.stdout.flush()
                with open(outpath+f"/{self.encoder_name}_{user_id}_pos.npy", "wb") as fi:
                    np.savez(fi, *windows)        
            # print(docs_tensor)        
        return users

    
class FastTextEncoder(Encoder):

    def __init__(self, pretrained_weights) -> None:
        super().__init__()
        self.encoder = fasttext.load_model(pretrained_weights)       
        self.encoder_name = "fasttext"

    def encode(self, inpath, outpath, window_size):
        inpath=inpath+"pkl/"        
        
        print("\n> FastText features")        
        
        users = []
        data = glob.glob(inpath+"/idx_*")
        
        with tqdm(data, unit="users") as pbar:
            for user_path in pbar:
                with open(user_path, "rb") as fi:
                    user_id, doc_lens, docs = pickle.load(fi) 
                pbar.set_description(f"user: {user_id} ({len(docs)})")
                users.append(user_id)        
                encoded_words = []
                for doc in docs:
                    encoded_words += [self.encoder[w] for w in doc]

                #convert all tensors into a single mega tensor
                mega_tensor = np.stack(encoded_words)        
                #slice the mega tensor into slices of window_size
                # from ipdb import set_trace; set_trace()
                n_windows = math.ceil(mega_tensor.shape[0]/window_size)
                windows = []
                for j in range(n_windows):
                    batch_docs = mega_tensor[window_size*j:window_size*(j+1) , :]
                    windows.append(batch_docs)
                
                # sys.stdout.write("\r> features | user: {}".format(user_id))
                # sys.stdout.flush()
                with open(outpath+f"/{self.encoder_name}_{user_id}_pos.npy", "wb") as fi:
                    np.savez(fi, *windows)        
            # print(docs_tensor)        
        return users
        

class W2VEncoder(Encoder):
    def __init__(self, inpath, outpath, pretrained_weights, emb_encoding, 
                min_word_freq=5, max_vocab_size=None) -> None:
        super().__init__()
        self.vocab = None
        self.inpath = inpath
        self.outpath = outpath
        self.pretrained_weights = pretrained_weights
        self.emb_encoding = emb_encoding        
        self.min_word_freq = min_word_freq
        self.max_vocab_size = max_vocab_size
        self.encoder_name = "w2v"

    def doc2idx(self, doc):
        
        doc = doc.split(" ")            
        try:
            doc_idx = [self.vocab[w] for w in doc if w in self.vocab]		
        except TypeError:
            self.vocab = self.get_vocab()    
            doc_idx = [self.vocab[w] for w in doc if w in self.vocab]		
        doc_len = len(doc_idx)	    
        
        return doc_len, np.array(doc_idx, dtype=np.int32) 

    def encode(self, inpath, outpath, window_size):
        inpath=inpath+"pkl/"
        print("\n> word2vec features")    
        with open(inpath+"word_emb.npy", "rb") as f:
            E = np.load(f)        
        users = []
        for user_path in glob.glob(inpath+"/idx_*"):
            with open(user_path, "rb") as fi:
                user_id, doc_lens, docs_idx = pickle.load(fi) 
            
            users.append(user_id)
            #concatenate all documents into a  single vector
            docs = np.concatenate(docs_idx, axis=None)                
            # from ipdb import set_trace; set_trace()
            n_windows = math.ceil(len(docs)/window_size)                
            windows = []
            for i in range(n_windows):
                window = docs[i*window_size:(i+1)*window_size] 
                try:
                    E_window = E[:, window].T
                    windows.append(E_window)
                except:
                    from ipdb import set_trace; set_trace()
            with open(outpath+f"/{self.encoder_name}_{user_id}_pos.npy", "wb") as fi:
                np.savez(fi, *windows)        
            sys.stdout.write("\r> features | user: {}".format(user_id))
            sys.stdout.flush()
        return users    

    def get_vocab(self):
        
        pkl_path=self.outpath+"pkl/"
        vocab = None
        word_counts = None
        if not os.path.exists(os.path.dirname(pkl_path)):
            os.makedirs(os.path.dirname(pkl_path)) 

        try:
            with open(pkl_path+"vocab.pkl", "rb") as fi:        
                vocab, word_counts = pickle.load(fi)        
                print("[found cached vocabulary]")
        except FileNotFoundError:
            pass

        if not vocab:
            #compute vocabulary
            vocab, word_counts = extract_vocabulary(self.inpath, 
                                                    min_word_freq=self.min_word_freq,
                                                    max_vocab_size=self.max_vocab_size)        
            vocab_len = len(vocab)
            #extract word embeddings
            E, vocab_redux = extract_word_embeddings(self.pretrained_weights, vocab, 
                                                    encoding=self.emb_encoding)
            #vocab_redux has only words for which an embedding was found
            print("[vocab size: {} > {}]".format(vocab_len,len(vocab_redux)))
            vocab = vocab_redux
            
            with open(pkl_path+"word_emb.npy", "wb") as f:
                np.save(f, E)    
            with open(pkl_path+"vocab.pkl", "wb") as f:
                pickle.dump([vocab_redux, word_counts], f, pickle.HIGHEST_PROTOCOL)
        
        return vocab
 
def extract_vocabulary(inpath, min_word_freq=5, max_vocab_size=None):    
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
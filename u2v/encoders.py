import tadat.core as core
import numpy as np

def encode(docs, max_size, encoder, encoder_args):
    if encoder == "BOW":
        X = BOW_encoder(docs, encoder_args)    
    elif encoder == "W2V":
        X = word2vec_encoder(docs, max_size, encoder_args)
    elif encoder == "BERT":
        X = BERT_encoder(docs, max_size, encoder_args)        
    else:
        raise NotImplementedError
    return X


def BOW_encoder(docs, args):
    weights = args["weights"]
    sparse  = args["sparse"]
    vocab_size  = args["vocab_size"]
    if weights == "binary":
        X = core.features.BOW(docs, vocab_size, sparse=sparse)
    elif weights == "frequency":
        X = core.features.BOW_freq(docs, vocab_size, sparse=sparse)

def word2vec_agg_encoder(docs, agg, E):    
    raise NotImplementedError
    n_docs = len(docs)
    emb_size = E.shape[0]    
    #+1 for an extra embedding for the entire sentence (the first position)
    X = np.zeros((n_docs, max_doc_size+1, emb_size))
    if agg == "bin":
        for i,doc in enumerate(docs):        
            word_embeddings = E[:, np.unique(doc)].T
            sentence_embedding = word_embeddings.sum(axis=0)    
            # from ipdb import set_trace; set_trace()
            doc_len = word_embeddings.shape[0]
            X[i, 0] = sentence_embedding
            X[i, 1:(1+doc_len)] = word_embeddings
    elif agg == "sum":
        for i,doc in enumerate(docs):        
            word_embeddings = E[:, doc].T
            sentence_embedding = word_embeddings.sum(axis=0)    
            X[i, 0] = sentence_embedding
            X[i, 1:] = word_embeddings
    else:
        raise NotImplementedError
    return X

def word2vec_encoder(docs, E):    

    n_docs = len(docs)
    emb_size = E.shape[0]    
    #+1 for an extra embedding for the entire sentence (the first position)
    X = np.zeros((n_docs, emb_size))    
    for i,doc in enumerate(docs):        
        word_embeddings = E[:, docs].T
        sentence_embedding = word_embeddings.sum(axis=0)    
        # from ipdb import set_trace; set_trace()
        doc_len = word_embeddings.shape[0]
        X[i, 0] = sentence_embedding
        X[i, 1:(1+doc_len)] = word_embeddings
    
    else:
        raise NotImplementedError
    return X

def BERT_encoder(docs, encoder_args):
    raise NotImplementedError


class Word2VecEncoder():

    def __init__(self, embeddings_path, m):
        pass
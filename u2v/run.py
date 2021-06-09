import argparse
import os
from u2v.lib import build_data, train_model, encode_users
from u2v.encoders import BERTEncoder, W2VEncoder, ELMoEncoder, FastTextEncoder
import torch 
import pprint
import json

def cmdline_args():
    parser = argparse.ArgumentParser(description="Train User2Vec")
    
    parser.add_argument('-conf_path', type=str, required=True, help='config json file')    
    parser.add_argument('-docs', type=str, help='documents file')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')    
    parser.add_argument('-train', action="store_true", 
                        help='train mode (assumes data was already extracted)')
    parser.add_argument('-build', action="store_true", 
                        help='build training data (does not train)')
    parser.add_argument('-encode', action="store_true", 
                        help='encode training data (does not train)')
    parser.add_argument('-cache', action="store_true", 
                        help='read from cached if data was already processed')    
    parser.add_argument('-device', type=str, default="auto", help='device')
    
    
    
    return parser.parse_args()	


if __name__ == "__main__" :    
    args = cmdline_args()    
    print("** U2V **")
    pprint.pprint(vars(args))
    
    print("")
    device = None
    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)

    with open(args.conf_path) as fi:
            conf = json.load(fi)
    print(" > confs: ")
    pprint.pprint(conf)
    encoder_type = conf["encoder_type"]

    output_path=f"{args.output}/{encoder_type}/"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path)) 
    

    if encoder_type == "bert":
        encoder = BERTEncoder(pretrained_weights=conf["pretrained_weights"], 
                              encoder_batchsize=conf.get("encoder_batch_size", 128), 
                              device=device)
        
    elif encoder_type == "elmo":
        encoder = ELMoEncoder(pretrained_weights=conf["pretrained_weights"], 
                              encoder_batchsize=conf.get("encoder_batch_size", 128), 
                              device=device)

    elif encoder_type == "fasttext":
        encoder = FastTextEncoder(pretrained_weights=conf["pretrained_weights"] )
    
    elif encoder_type == "w2v":
        encoder = W2VEncoder(inpath=args.docs, outpath=output_path, 
                            embeddings_path=conf["pretrained_weights"], 
                            emb_encoding=conf.get("pretrained_weights","utf8"),
                            min_word_freq=conf.get("min_word_freq", 1),
                            max_vocab_size=conf.get("vocab_size"))
    else:
        raise NotImplementedError
    
    if (not args.train and not args.encode) or args.build:
        print("> prepare data")
        build_data(args.docs, output_path, encoder, 
                random_seed=conf["seed"], 
                min_docs_user=conf.get("min_docs_user", 1),
                max_docs_user=conf.get("max_docs_user"), 
                reset=args.reset)
    
    if (not args.train and not args.build) or args.encode:
        print("> encode data")        
        encode_users(path=output_path, encoder=encoder,
                    window_size=conf["window_size"], cache=args.cache)        

    if (not args.build and not args.encode) or args.train:
        print("> train")
        #get configs
        run_id = conf.get("run_id","auto")
        epochs = conf.get("epochs", 5)
        margin = conf.get("margin", 1)
        initial_lr = conf.get("lr", 0.01)
        batch_size = conf.get("batch_size", 1)
        val_split = conf.get("val_split", 0.2)

        if run_id == "auto":            
            run_id = f"{epochs}_{margin}_{initial_lr}"
        #create tensorboard log
        logs_path = f"{args.output}/logs/{encoder_type}/{run_id}/"        
        if not os.path.exists(os.path.dirname(logs_path)):
            os.makedirs(os.path.dirname(logs_path)) 

        train_model(output_path, encoder, 
                    run_id=run_id,
                    epochs=epochs,
                    initial_lr=initial_lr,
                    margin=margin,                    
                    batch_size=batch_size,
                    validation_split=val_split,
                    logs_path=logs_path,
                    cache=args.cache,                    
                    device=device)

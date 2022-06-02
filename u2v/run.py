import argparse
import os
from u2v.lib import build_data, train_model, encode_users
from u2v.encoders import BERTEncoder, ClinicalBertEncoder, W2VEncoder, ELMoEncoder, FastTextEncoder
import torch 
import pprint
import json

def cmdline_args():
    parser = argparse.ArgumentParser(description="Train User2Vec")
    
    parser.add_argument('-conf_path', type=str, help='config json file')    
    parser.add_argument('-docs', type=str, required=True, help='documents file')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')    
    parser.add_argument('-train', action="store_true", 
                        help='train mode (assumes data was already encoded)')
    parser.add_argument('-build', action="store_true", 
                        help='build training data (does not train)')
    parser.add_argument('-encode', action="store_true", 
                        help='encode training data (does not train)')    
    parser.add_argument('-reset', action="store_true", 
                        help='re-build training data')    
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
    default_conf = {
        "encoder_type":"elmo",
        "seed" : 426, 
        "pretrained_weights" : "small",
        "encoder_batch_size" : 256,        
        "window_size" : 128
    }    

    if args.conf_path:
        with open(args.conf_path) as fi:
                conf = json.load(fi)
    else:
        conf = default_conf

    print(" > confs: ")
    pprint.pprint(conf)

    encoder_type = conf["encoder_type"]
    output_path=f"{args.output}/{encoder_type}/"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path)) 
    
    #default batch size: 128
    if encoder_type == "bert":
        encoder = BERTEncoder(pretrained_weights=conf["pretrained_weights"], 
                              encoder_batchsize=conf.get("encoder_batch_size", 128), 
                              device=device)
    elif encoder_type == "clinicalbert":
        encoder = ClinicalBertEncoder(
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
    
    if args.build:
        print("> build")
        build_data(args.docs, output_path, encoder, 
                random_seed=conf["seed"], 
                min_docs_user=conf.get("min_docs_user", 10),
                max_docs_user=conf.get("max_docs_user"), 
                reset=args.reset)
    
    if args.encode:
        print("> encode")        
        encode_users(path=output_path, encoder=encoder,
                    window_size=conf["window_size"], reset=args.reset)        

    if args.train:
        print("> train")
        #get configs
        run_id = conf.get("run_id","auto")
        epochs = conf.get("epochs", 5)
        margin = conf.get("margin", 1)
        initial_lr = conf.get("lr", 0.01)
        batch_size = conf.get("batch_size", 32)
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
                    reset=args.reset,                    
                    device=device)

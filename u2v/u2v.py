import argparse
import os
from lib import build_data, train_model, encode_users
import torch 

def cmdline_args():
    parser = argparse.ArgumentParser(description="Train User2Vec")
    parser.add_argument('-input', type=str, required=True, help='input folder')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')    
    parser.add_argument('-emb', type=str, required=True, help='path to word embeddings')    
    parser.add_argument('-vocab_size', type=int, help='max size of vocabulary')
    parser.add_argument('-min_word_freq', type=int, help='ignore words that occur less than min_word_freq times',default=5)
    parser.add_argument('-min_docs_user', type=int, help='ignore users with less min_docs_user documents',default=3)
    parser.add_argument('-seed', type=int, default=1234, help='random number generator seed')    
    parser.add_argument('-neg_samples', type=int, help='number of negative samples', default=10)
    parser.add_argument('-epochs', type=int, default=20, help='number of training epochs')    
    parser.add_argument('-lr', type=float, default=0.0005, help='learning rate')    
    parser.add_argument('-margin', type=float, default=1, help='hinge loss margin')    
    parser.add_argument('-reset', action="store_true", help='reset embeddings that were already computed')
    parser.add_argument('-train', action="store_true", help='train mode (assumes data was already extracted)')
    parser.add_argument('-build', action="store_true", help='build training data (does not train)')
    parser.add_argument('-encode', action="store_true", help='encode training data (does not train)')
    parser.add_argument('-reset_cache', action="store_true", help='reset cached users (i.e. all are trained)')
    parser.add_argument('-encoder_type', choices=["w2v","bert"], default="w2v", help="encoder type")
    parser.add_argument('-device', type=str, default="auto", help='device')
    return parser.parse_args()	

if __name__ == "__main__" :    
    args = cmdline_args()
    info = "[input: {} | output: {} | embedding: {} | epochs: {} | lr: {} | margin: {} | neg samples: {} | reset: {} | device: {}]"
    print(info.format(os.path.basename(args.input), 
                                        args.output,
                                        os.path.basename(args.emb),
                                        args.epochs,
                                        args.lr,
                                        args.margin,
                                        args.neg_samples,
                                        args.reset,
                                        args.device))   

    if (not args.train and not args.encode) or args.build:
        print("> prepare data")
        build_data(args.input, args.output, args.encoder_type, args.emb, emb_encoding="latin-1", min_word_freq=args.min_word_freq, max_vocab_size=None, random_seed=args.seed, min_docs_user=args.min_docs_user, reset=args.reset)
    
    if (not args.train and not args.build) or args.encode:
        print("> encode data")
        encode_users(args.output, args.encoder_type)
        # if args.encoder_type == "w2v":
        #     word2vec_features(args.output)

    if (not args.build and not args.encode) or args.train:
        device = None
        if args.device == "auto":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device(args.device)
        print("> train")
        train_model(args.output, epochs=args.epochs, initial_lr=args.lr, 
                    margin=args.margin,
                    reset=args.reset_cache,
                    device=device)

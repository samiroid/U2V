import argparse
import os
from u2v.lib import build_data, train_model, encode_users
from u2v.encoders import BERTEncoder, W2VEncoder, ELMoEncoder, FastTextEncoder
import torch 

def cmdline_args():
    parser = argparse.ArgumentParser(description="Train User2Vec")
    parser.add_argument('-input', type=str, required=True, help='input folder')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')    
    parser.add_argument('-emb', type=str, required=True, help='path to word embeddings')    
    parser.add_argument('-emb_enc', type=str,  default="utf-8", 
                        help='word embeddings text file encoding')    
    parser.add_argument('-vocab_size', type=int, help='max size of vocabulary')
    parser.add_argument('-min_word_freq', type=int, default=5,
                        help='ignore words that occur less than min_word_freq times'
                        )
    parser.add_argument('-min_docs_user', type=int, default=3,
                        help='ignore users with less min_docs_user documents')
    parser.add_argument('-seed', type=int, default=1234, 
                        help='random number generator seed')    
    # parser.add_argument('-neg_samples', type=int, help='number of negative samples',
    #                      default=10)
    parser.add_argument('-epochs', type=int, default=20, help='number of training epochs')    
    parser.add_argument('-val_split', type=float, default=0.2, help='percentage of data reserved for validation')    
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')    
    parser.add_argument('-margin', type=int, default=1, help='hinge loss margin')    
    parser.add_argument('-reset', action="store_true", 
                        help='reset embeddings that were already computed')
    parser.add_argument('-train', action="store_true", 
                        help='train mode (assumes data was already extracted)')
    parser.add_argument('-build', action="store_true", 
                        help='build training data (does not train)')
    parser.add_argument('-encode', action="store_true", 
                        help='encode training data (does not train)')
    parser.add_argument('-reset_cache', action="store_true", 
                        help='reset cached users (i.e. all are trained)')
    parser.add_argument('-encoder_type', choices=["w2v","bert","elmo","fasttext"], default="w2v", 
                        help="encoder type")
    parser.add_argument('-device', type=str, default="auto", help='device')
    parser.add_argument('-run_id', type=str, default="", help='run id')
    parser.add_argument('-pretrained_model', type=str, default="bert-base-uncased", 
                        help='name of pretrained transformer weights')
    parser.add_argument('-encoder_batchsize', type=int, default=80, 
                        help='encoder batch size')
    parser.add_argument('-batch_size', type=int, default=128, 
                        help='batch size')
    parser.add_argument('-window_size', type=int, default=128, help='window size')
    return parser.parse_args()	


if __name__ == "__main__" :    
    args = cmdline_args()
    info = "[encoder: {} | input: {} | output: {} | reset: {} | device: {}]"
    print(info.format(args.encoder_type, os.path.basename(args.input), 
                        args.output,                        
                        args.reset,
                        args.device))   
    output_path=f"{args.output}/{args.encoder_type}/"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path)) 

    if args.encoder_type == "bert":
        encoder = BERTEncoder(pretrained_model=args.pretrained_model, 
                    encoder_batchsize=args.encoder_batchsize, device=args.device)
        
    elif args.encoder_type == "w2v":
        encoder = W2VEncoder(inpath=args.input, outpath=output_path, 
                            embeddings_path=args.emb, emb_encoding=args.emb_enc, 
                            min_word_freq=args.min_word_freq, 
                            max_vocab_size=args.vocab_size)
    elif args.encoder_type == "elmo":
        encoder = ELMoEncoder(device=args.device,encoder_batchsize=args.encoder_batchsize,)
    elif args.encoder_type == "fasttext":
        encoder = FastTextEncoder(embeddings_path=args.emb)
    else:
        raise NotImplementedError
    
    if (not args.train and not args.encode) or args.build:
        print("> prepare data")
        build_data(args.input, output_path, encoder, 
                random_seed=args.seed, min_docs_user=args.min_docs_user, 
                reset=args.reset)
    
    if (not args.train and not args.build) or args.encode:
        print("> encode data")        
        encode_users(path=output_path, encoder=encoder,
                    window_size=args.window_size )        

    if (not args.build and not args.encode) or args.train:
        device = None
        if args.device == "auto":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device(args.device)
        if args.run_id == "auto":            
            run_id = f"{args.epochs}_{args.margin}_{args.lr}"
            # from ipdb import set_trace; set_trace()
        else:
            run_id = args.run_id
        print("> train")
        # from ipdb import set_trace;set_trace()
        train_model(output_path, run_id=run_id, batch_size=args.batch_size,
                    epochs=args.epochs, initial_lr=args.lr, margin=args.margin,
                    validation_split = args.val_split, reset=args.reset_cache,
                    device=device)

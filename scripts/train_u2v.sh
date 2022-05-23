BASE_PATH="/Users/samir/Dev/projects/user2vec/U2V/U2V/"
WORD_EMBEDDINGS=$BASE_PATH"/DATA/word_embeddings.txt"
CORPUS=$BASE_PATH"/DATA/small_user_tweets.txt"
OUTPUT_PATH=$BASE_PATH"/DATA/out/"

python u2v/u2v.py -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
                        -lr 10 \
                        -epochs 20 \
                        -neg_samples 5 \
                        -margin 5 \
                        -min_word_freq 1 \
                        -encoder_type bert \
                        -encode \
                        # -build \
                        # -reset_cache \
                        # -reset \
                        # -reset
                        # -build \
                        # -reset \
                        # -train
                        

BASE_PATH="/Users/samir/Dev/projects/user2vec/U2V/U2V/"
BASE_PATH="/home/silvio/home/projects/U2V/"
WORD_EMBEDDINGS=$BASE_PATH"/DATA/word_embeddings.txt"
WORD_EMBEDDINGS=$BASE_PATH"/DATA/fasttext_twitter_raw.bin"
CORPUS=$BASE_PATH"/DATA/small_user_tweets.txt"
OUTPUT_PATH=$BASE_PATH"/DATA/out/"

python -m u2v.run -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
                        -emb_enc "latin-1" \
                        -lr 10 \
                        -epochs 20 \
                        -margin 2 \
                        -min_word_freq 1 \
                        -device cuda:1 \
                        -reset_cache \
                        -val_split 0.2 \
                        -batch_size 128 \
                        -encoder_batchsize 128 \
                        -encoder_type bert \
                        -build \
                        -encode \
                        # -run_id auto \
                        # -train
                        # -reset \
                        # -train
                        # -reset
                        # -build \
                        # -reset \


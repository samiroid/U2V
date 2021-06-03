BASE_PATH="/Users/samir/Dev/projects/user2vec/U2V/U2V/"
BASE_PATH="/home/silvio/home/projects/U2V/"
WORD_EMBEDDINGS=$BASE_PATH"/DATA/word_embeddings.txt"
WORD_EMBEDDINGS=$BASE_PATH"/DATA/fasttext_twitter_raw.bin"
CORPUS=$BASE_PATH"/DATA/small_user_tweets.txt"
OUTPUT_PATH=$BASE_PATH"/DATA/out/"


LR=$1
MARGIN=$2

python -m u2v.run -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
                        -emb_enc "latin-1" \
                        -min_word_freq 1 \
                        -device cpu \
                        -encoder_batchsize 128 \
                        -batch_size 128 \
                        -val_split 0.2 \
                        -epochs 10 \
                        -encoder_type elmo \
                        -lr $LR \
                        -margin $MARGIN \
                        -run_id auto \
                        -train 
                        # -reset_cache \

# python -m u2v.run -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
#                         -emb_enc "latin-1" \
#                         -min_word_freq 1 \
#                         -device cuda:1 \
#                         -encoder_batchsize 128 \
#                         -reset_cache \
#                         -batch_size 128 \
#                         -val_split 0.2 \
#                         -epochs 10 \
#                         -encoder_type bert \
#                         -lr $LR \
#                         -margin $MARGIN \
#                         -run_id auto \
#                         -train &

# python -m u2v.run -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
#                         -emb_enc "latin-1" \
#                         -min_word_freq 1 \
#                         -device cuda:3 \
#                         -encoder_batchsize 128 \
#                         -reset_cache \
#                         -batch_size 128 \
#                         -val_split 0.2 \
#                         -epochs 10 \
#                         -encoder_type fasttext \
#                         -lr $LR \
#                         -margin $MARGIN \
#                         -run_id auto \
#                         -train
                        

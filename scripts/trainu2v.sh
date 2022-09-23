BASE_PATH="/Users/silvioamir/Dev/projects/user2vec/U2V/U2V"
# BASE_PATH="/home/silvio/home/projects/U2V/"
WORD_EMBEDDINGS=$BASE_PATH"/DATA/word_embeddings.txt"

WORD_EMBEDDINGS=$BASE_PATH"/DATA/fasttext_twitter_raw.bin"
CORPUS=$BASE_PATH"/DATA/small_user_tweets.txt"
OUTPUT_PATH=$BASE_PATH"/DATA/out/"
CONF=$BASE_PATH"/confs/train/fasttext.json"
python -m u2v.run -docs $CORPUS \
                -output $OUTPUT_PATH \
                -conf_path $CONF \
                -build \
                -encode \
                -train \
                -device cpu
                        # -emb_enc "latin-1" \
                        # -lr 10 \
                        # -epochs 20 \
                        # -margin 2 \
                        # -min_word_freq 1 \
                        # -reset_cache \
                        # -val_split 0.2 \
                        # -batch_size 128 \
                        # -encoder_batchsize 64 \
                        # -encoder_type elmo \
                        # -train
                        # -device cuda:1 \
                        # -run_id auto \
                        # -train

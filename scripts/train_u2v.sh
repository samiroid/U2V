BASE_PATH="/Users/samir/Dev/projects/user2vec/U2V/U2V/"
WORD_EMBEDDINGS=$BASE_PATH"/DATA/word_embeddings.txt"
CORPUS=$BASE_PATH"/DATA/sample.txt"
OUTPUT_PATH=$BASE_PATH"/DATA/out/"

python u2v/u2v.py -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
                        -lr 0.1 \
                        -epochs 20 \
                        -neg_samples 2 \
                        -margin 2 \
                        -min_word_freq 1 \
                        # -train \
                        # -reset_cache \
                        # -encode \
                        # -build \
                        # -reset \
                        # -train
                        

BASE_PATH="/Users/silvioamir/Dev/projects/user2vec/U2V/U2V"
CORPUS=$BASE_PATH"/samples/data.txt"
CORPUS=$BASE_PATH"/DATA/small_user_tweets.txt"
CONF=$BASE_PATH"/samples/conf.json"
OUTPUT_PATH=$BASE_PATH"/DATA/out/"
python -m u2v.run -docs $CORPUS \
                -output $OUTPUT_PATH \
                -conf_path $CONF \
                -device cpu \
                -train \
                -encode \
                -build \
                -reset 


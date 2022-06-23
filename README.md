# U2V 

User2Vec is a neural model that learns static user representations based on a collections of texts. This is a repo is re-implementation of the [original model](https://github.com/samiroid/usr2vec) in pytorch and with support for more modern text encoders (i.e. [Fasttext](https://fasttext.cc/docs/en/english-vectors.html), [ELMo](https://allenai.org/allennlp/software/elmo) and [Transformers](https://huggingface.co/docs/transformers/index)). Refer to the [publications](#citations) for details on the model.

### Install

User2Vec can be installed directly from the repository or manually after downloading the code  
- **repository**: `pip install git+https://github.com/samiroid/U2V.git`
- **manually**: `pip install [PATH TO CODE]` or `python [PATH TO CODE]/setup.py install` 

### Run

User2Vec can be run as a pipeline with 3 steps:
1. build: Preprocess and vectorize documents
2. encode: Encode tokens with pretrained word embedding representations
3. train: Train user2vec model to learn user embedding representations

This allows for faster experimentation with different configurations. To run the pipeline on the included sample dataset with default configuration use

`python -m u2v.run -docs DATA/sample.txt  -output [OUTPUT_PATH]  -conf_path confs/sample.json -build -encode -train -device cpu`

User2Vec can be configured with the following parameters for the **pipeline**, **encoder**, and **model** (default values in parentheses)

**Pipeline Parameters**

- `-conf_path`: path to config file
- `-docs`: path to input documents
- `-output`: path to output folder
- `-device (cpu)`: device (cpu or cuda)
- `-train`: train model
- `-build`: buid training data
- `-encode`: encode word tokens
- `-reset`: rebuild training data

**Encoder Parameters**:

- `encoder_type (elmo)`: encoder type (elmo, fasttext, bert, clinicalbert)
- `seed (426)`: random seed 
- `pretrained_weights (small)`: pretrained weights (**ELMO:** {*small, medium, original*}; [BERT](https://huggingface.co/models), **FastText**: *[path to embeddings]*)
- `encoder_batch_size (256)`: batch size for the encoder module (relevant when using GPU)
- `max_docs_user (10)`: maximum number of documents per user
- `window_size (128)`: window size 

**Model Parameters**:

- `margin (1)`: hinge loss margin
- `batch_size (32)`: batch size
- `initial_lr (0.01)`: initial learning rate
- `validation_split (0.2)`: percentage of validation data
- `epochs (5)`: number of epochs 
- `run_id`: identifier for the run (if not set and id will be generated as `[MARGIN]_[INITIAL_LR]_[EPOCHS]`)

Note that the choice of model parameters can have a significant impact on the performance of the downstream applications. We recommend experimenting with different learning rates and margins.

### Output

The output of the pipeline is stored at `[OUTPUT_PATH]/[ENCODER_TYPE]` and consists of two sets of representations:

- `U_[ENCODER_TYPE]_[RUN_ID].txt`: User2vec user embeddings learned with `[ENCODER_TYPE]` word representations and run ID `[RUN_ID]` 

- `U_[ENCODER_TYPE]_mean.txt`: user embeddings computed as the average of word representations 

### Citations
If you use this code please cite one of the following papers:
> Irani, D., Wrat, A., and Amir, S., 2021. *Early Detection of Online Hate Speech Spreaders with Learned User Representations*. In CLEF 2021 Labs and Workshops, Notebook Papers 2021.

> Amir, S., Coppersmith, G., Carvalho, P., Silva, M.J. and Wallace, B.C., 2017. *Quantifying Mental Health from Social Media with Neural User Embeddings*. In Journal of Machine Learning Research, W&C Track, Volume 68. 

"""Train a sequence to sequence model.

This script is sourced from Siraj Rival
https://github.com/llSourcell/How_to_make_a_text_summarizer/blob/master/train.ipynb
"""
import os
import time
import _pickle as pickle
import random
import argparse

import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils

from keras.regularizers import l2
from keras.callbacks import TensorBoard

import config
from constants import empty, eos
from sample_gen import vocab_fold, lpadd, gensamples
from utils import prt, load_embedding, process_vocab, load_split_data
from model import create_model, inspect_model

# you should use GPU...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ...but if it is busy then you always can fall back to your CPU with
# os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--FN0', default='vocabulary-embedding', help="filename of vocab embeddings")
parser.add_argument('--FN1', default='train', help="filename of model weights")
parser.add_argument('--batch-size', type=int, default=32, help='input batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--maxlend', type=int, default=100, help='max length of description')
parser.add_argument('--maxlenh', type=int, default=15, help='max length of head')
parser.add_argument('--rnn-size', type=int, default=512, help='size of RNN layers')
parser.add_argument('--rnn-layers', type=int, default=3, help='number of RNN layers')
parser.add_argument('--nsamples', type=int, default=640, help='number of samples per epoch')
parser.add_argument('--nflips', type=int, default=0, help='number of flips')
parser.add_argument('--temperature', type=float, default=.8, help='RNN temperature')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
args = parser.parse_args()

# static variables
FN = 'train'
maxlend = args.maxlend
maxlenh = args.maxlenh
maxlen = maxlend + maxlenh
rnn_size = args.rnn_size
rnn_layers = args.rnn_layers
activation_rnn_size = 40 if maxlend else 0
LR = args.lr
batch_size = args.batch_size
nb_unknown_words = 10

# training parameters
seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
nb_train_samples = np.int(np.floor(args.nsamples / batch_size)) * batch_size  # num training samples
nb_val_samples = nb_train_samples  # num validation samples
optimizer = 'adam'
regularizer = l2(weight_decay) if weight_decay else None

# seed weight initialization
random.seed(seed)
np.random.seed(seed)

embedding, idx2word, word2idx, glove_idx2idx = load_embedding(args.FN0, nb_unknown_words)
vocab_size, embedding_size = embedding.shape
oov0 = vocab_size - nb_unknown_words
idx2word = process_vocab(idx2word, vocab_size, oov0, nb_unknown_words)
X_train, X_test, Y_train, Y_test = load_split_data(args.FN0, nb_val_samples, seed)

# print a sample recipe to make sure everything looks right
print('Random head, description:')
i = 811
prt('H', Y_train[i], idx2word)
prt('D', X_train[i], idx2word)

model = create_model(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    maxlen=maxlen,
    maxlend=maxlend,
    maxlenh=maxlenh,
    regularizer=regularizer,
    optimizer=optimizer,
    LR=LR,
    p_emb=p_emb,
    embedding=embedding,
    rnn_layers=rnn_layers,
    activation_rnn_size=activation_rnn_size,
    rnn_size=rnn_size,
    p_W=p_W,
    p_U=p_U,
    p_dense=p_dense,
)
inspect_model(model)

# load pre-trained model weights
FN1_filename = os.path.join(config.path_models, '{}.hdf5'.format(args.FN1))
if args.FN1 and os.path.exists(FN1_filename):
    model.load_weights(FN1_filename)
    print('Model weights loaded from {}'.format(FN1_filename))

# print samples before training
gensamples(
    skips=2,
    k=10,
    batch_size=batch_size,
    short=False,
    temperature=args.temperature,
    use_unk=True,
    model=model,
    sequence=sequence,
    data=(X_test, Y_test),
    idx2word=idx2word,
    maxlen=maxlen,
    maxlenh=maxlenh,
    maxlend=maxlend,
    oov0=oov0,
    glove_idx2idx=glove_idx2idx,
    vocab_size=vocab_size,
    nb_unknown_words=nb_unknown_words,
)

# Data generator

"""Data generator generates batches of inputs and outputs/labels for training. The inputs are each made from two parts. The first maxlend words are the original description, followed by `eos` followed by the headline which we want to predict, except for the last word in the headline which is always `eos` and then `empty` padding until `maxlen` words.

For each, input, the output is the headline words (without the start `eos` but with the ending `eos`) padded with `empty` words up to `maxlenh` words. The output is also expanded to be y-hot encoding of each word.

To be more realistic, the second part of the input should be the result of generation and not the original headline.
Instead we will flip just `nflips` words to be from the generator, but even this is too hard and instead
implement flipping in a naive way (which consumes less time.) Using the full input (description + eos + headline) generate predictions for outputs. For nflips random words from the output, replace the original word with the word with highest probability from the prediction.
"""


def flip_headline(x, nflips=None, model=None, debug=False):
    """Flip some of the words in the second half (headline) with words predicted by the model."""
    if nflips is None or model is None or nflips <= 0:
        return x

    batch_size = len(x)
    assert np.all(x[:, maxlend] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
        flips = sorted(random.sample(range(maxlend + 1, maxlen), nflips))
        if debug and b < debug:
            print(b)
        for input_idx in flips:
            if x[b, input_idx] == empty or x[b, input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlend + 1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = oov0
            if debug and b < debug:
                print('{} => {}'.format(idx2word[x_out[b, input_idx]], idx2word[w]),)
            x_out[b, input_idx] = w
        if debug and b < debug:
            print()
    return x_out


def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    """Convert description and hedlines to padded input vectors; headlines are one-hot to label."""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [
        vocab_fold(lpadd(xd, maxlend=maxlend, eos=eos) + xh, oov0, glove_idx2idx, vocab_size, nb_unknown_words)
        for xd, xh in zip(xds, xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug)

    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh, oov0, glove_idx2idx, vocab_size, nb_unknown_words) + [eos] + [empty] * maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i, :, :] = np_utils.to_categorical(xh, vocab_size)

    return x, y


def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
    """Yield batches.

    for training use nb_batches=None
    for validation generate deterministic results repeating every nb_batches
    """
    # while training it is good idea to flip once in a while the values of the headlines from the
    # value taken from Xh to value generated by the model.
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, 2e10)
        random.seed(c + 123456789 + seed)
        for b in range(batch_size):
            t = random.randint(0, len(Xd) - 1)

            xd = Xd[t]
            s = random.randint(min(maxlend, len(xd)), max(maxlend, len(xd)))
            xds.append(xd[:s])

            xh = Xh[t]
            s = random.randint(min(maxlenh, len(xh)), max(maxlenh, len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c += 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)


def test_gen(gen, n=5):
    """Generate test batches."""
    Xtr, Ytr = next(gen)
    for i in range(n):
        assert Xtr[i, maxlend] == eos
        x = Xtr[i, :maxlend]
        y = Xtr[i, maxlend:]
        yy = Ytr[i, :]
        yy = np.where(yy)[1]
        prt('L', yy, idx2word)
        prt('H', y, idx2word)
        if maxlend:
            prt('D', x)

r = next(gen(X_train, Y_train, batch_size=batch_size))
valgen = gen(X_test, Y_test, nb_batches=3, batch_size=batch_size)

# Train
history = {}
traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=args.nflips, model=model)
valgen = gen(X_test, Y_test, nb_batches=nb_val_samples // batch_size, batch_size=batch_size)

callbacks = [TensorBoard(
    log_dir=os.path.join(config.path_logs, str(time.time())),
    histogram_freq=2, write_graph=False, write_images=False)]

h = model.fit_generator(
    traingen, samples_per_epoch=nb_train_samples,
    nb_epoch=args.epochs, validation_data=valgen, nb_val_samples=nb_val_samples,
    callbacks=callbacks,
)
for k, v in h.history.items():
    history[k] = history.get(k, []) + v
with open(os.path.join(config.path_models, 'history.pkl'.format(FN)), 'wb') as fp:
    pickle.dump(history, fp, -1)
model.save_weights(FN1_filename, overwrite=True)

# print samples after training
gensamples(
    skips=2,
    k=10,
    batch_size=batch_size,
    short=False,
    temperature=args.temperature,
    use_unk=True,
    model=model,
    sequence=sequence,
    data=(X_test, Y_test),
    idx2word=idx2word,
    maxlen=maxlen,
    maxlenh=maxlenh,
    maxlend=maxlend,
    oov0=oov0,
    glove_idx2idx=glove_idx2idx,
    vocab_size=vocab_size,
    nb_unknown_words=nb_unknown_words,
)

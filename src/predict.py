"""Predict a title for a recipe."""
from os import path
import random
import json
import pickle
import h5py
import numpy as np
from utils import str_shape
import keras.backend as K
import argparse

from config import path_models, path_data
from constants import FN1, FN0, nb_unknown_words, eos
from model import create_model
from sample_gen import gensamples

# set seeds in random libraries
seed = 42
random.seed(seed)
np.random.seed(seed)


def load_weights(model, filepath):
    """Load all weights possible into model from filepath.

    This is a modified version of keras load_weights that loads as much as it can
    if there is a mismatch between file and model. It returns the weights
    of the first layer in which the mismatch has happened
    """
    print('Loading', filepath, 'to', model.name)
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            print(name)
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                try:
                    layer = model.get_layer(name=name)
                except:
                    layer = None
                if not layer:
                    print('failed to find layer', name, 'in model')
                    print('weights', ' '.join(str_shape(w) for w in weight_values))
                    print('stopping to load all other layers')
                    weight_values = [np.array(w) for w in weight_values]
                    break
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                weight_value_tuples += zip(symbolic_weights, weight_values)
                weight_values = None
        K.batch_set_value(weight_value_tuples)
    return weight_values


def main(sample_str=None):
    """Predict a title for a recipe."""
    # load model parameters used for training
    with open(path.join(path_models, 'model_params.json'), 'r') as f:
        model_params = json.load(f)

    # create placeholder model
    model = create_model(**model_params)

    # load weights from training run
    load_weights(model, path.join(path_models, '{}.hdf5'.format(FN1)))

    # load recipe titles and descriptions
    with open(path.join(path_data, 'vocabulary-embedding.data.pkl'), 'rb') as fp:
        X_data, Y_data = pickle.load(fp)

    # load vocabulary
    with open(path.join(path_data, '{}.pkl'.format(FN0)), 'rb') as fp:
        embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
    vocab_size, embedding_size = embedding.shape
    oov0 = vocab_size - nb_unknown_words

    if sample_str is None:
        # load random recipe description if none provided
        i = np.random.randint(len(X_data))
        sample_str = ''
        sample_title = ''
        for w in X_data[i]:
            sample_str += idx2word[w] + ' '
        for w in Y_data[i]:
            sample_title += idx2word[w] + ' '
        y = Y_data[i]
        print('Randomly sampled recipe:')
        print(sample_title)
        print(sample_str)
    else:
        sample_title = ''
        y = [eos]

    x = [word2idx[w.rstrip('^')] for w in sample_str.split()]

    samples = gensamples(
        skips=2,
        k=1,
        batch_size=2,
        short=False,
        temperature=1.,
        use_unk=True,
        model=model,
        data=(x, y),
        idx2word=idx2word,
        oov0=oov0,
        glove_idx2idx=glove_idx2idx,
        vocab_size=vocab_size,
        nb_unknown_words=nb_unknown_words,
    )

    headline = samples[0][0][len(samples[0][1]):]
    ' '.join(idx2word[w] for w in headline)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-str', type=str, default=None, help='Sample recipe description')
    args = parser.parse_args()
    main(sample_str=args.sample_str)

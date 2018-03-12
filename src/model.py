"""Create an LSTM model for recipe summarization."""
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
import keras.backend as K
import numpy as np

from utils import str_shape
from constants import maxlend, maxlenh, maxlen, activation_rnn_size, optimizer, p_W, p_U, p_dense, p_emb, regularizer


def inspect_model(model):
    """Print the structure of Keras `model`."""
    for i, l in enumerate(model.layers):
        print(i, 'cls={} name={}'.format(type(l).__name__, l.name))
        weights = l.get_weights()
        print_str = ''
        for weight in weights:
            print_str += str_shape(weight) + ' '
        print(print_str)
        print()


class SimpleContext(Lambda):
    """Class to implement `simple_context` method as a Keras layer."""

    def __init__(self, fn, rnn_size, **kwargs):
        """Initialize SimpleContext."""
        self.rnn_size = rnn_size
        super(SimpleContext, self).__init__(fn, **kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        """Compute mask of maxlend."""
        return input_mask[:, maxlend:]

    def get_output_shape_for(self, input_shape):
        """Get output shape for a given `input_shape`."""
        nb_samples = input_shape[0]
        n = 2 * (self.rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)


def create_model(vocab_size, embedding_size, LR, rnn_layers, rnn_size, embedding=None):
    """Construct and compile LSTM model."""
    # create a standard stacked LSTM
    if embedding is not None:
        embedding = [embedding]
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
                        W_regularizer=regularizer, dropout=p_emb, weights=embedding, mask_zero=True,
                        name='embedding_1'))
    for i in range(rnn_layers):
        lstm = LSTM(rnn_size, return_sequences=True,
                    W_regularizer=regularizer, U_regularizer=regularizer,
                    b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
                    name='lstm_{}'.format(i + 1))
        model.add(lstm)
        model.add(Dropout(p_dense, name='dropout_{}'.format(i + 1)))

    def simple_context(X, mask, n=activation_rnn_size):
        """Reduce the input just to its headline part (second half).

        For each word in this part it concatenate the output of the previous layer (RNN)
        with a weighted average of the outputs of the description part.
        In this only the last `rnn_size - activation_rnn_size` are used from each output.
        The first `activation_rnn_size` output is used to computer the weights for the averaging.
        """
        desc, head = X[:, :maxlend, :], X[:, maxlend:, :]
        head_activations, head_words = head[:, :, :n], head[:, :, n:]
        desc_activations, desc_words = desc[:, :, :n], desc[:, :, n:]

        # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
        # activation for every head word and every desc word
        activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))
        # make sure we dont use description words that are masked out
        activation_energies = activation_energies + -1e20 * K.expand_dims(
            1. - K.cast(mask[:, :maxlend], 'float32'), 1)

        # for every head word compute weights for every desc word
        activation_energies = K.reshape(activation_energies, (-1, maxlend))
        activation_weights = K.softmax(activation_energies)
        activation_weights = K.reshape(activation_weights, (-1, maxlenh, maxlend))

        # for every head word compute weighted average of desc words
        desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
        return K.concatenate((desc_avg_word, head_words))

    if activation_rnn_size:
        model.add(SimpleContext(simple_context, rnn_size, name='simplecontext_1'))

    model.add(TimeDistributed(Dense(
        vocab_size,
        W_regularizer=regularizer,
        b_regularizer=regularizer,
        name='timedistributed_1')))
    model.add(Activation('softmax', name='activation_1'))

    # opt = Adam(lr=LR)  # keep calm and reduce learning rate
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    K.set_value(model.optimizer.lr, np.float32(LR))
    return model

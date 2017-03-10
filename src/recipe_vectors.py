from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import argparse
import pickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import config
import prep_data
from parse_ingredients import parse_ingredient_listlist
from utils import get_flat_ingredients_list, join_ingredients, section_print
from viz import plot_reduced

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def build_dataset(ingredients, vocab_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(
        ingredients).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in ingredients:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def get_ingredient_to_recipe(unique_ingredients, ingredients_joined):
    ingredient_to_recipe = {ingredient: [] for ingredient in unique_ingredients}
    for ingredient in unique_ingredients:
        for r, recipe in enumerate(ingredients_joined):
            if ingredient in recipe:
                ingredient_to_recipe[ingredient].append(r)
    return ingredient_to_recipe

def get_sample_pair(data, dictionary, reverse_dictionary, ingredient_to_recipe, ingredients):
    """Return a random pair of ingredients
    from a randomly selected recipe
    """
    ingredient_num = random.sample(data, 1)[0]
    ingredient = reverse_dictionary[ingredient_num]
    relevant_recipes = ingredient_to_recipe[ingredient]
    try:
        # try randomly sampling two ingredients from the recipe
        recipe = random.sample(relevant_recipes, 1)[0]
        ingredient_pairs = ingredients[recipe].copy()
        ingredient_pairs.remove(ingredient)
        try:
            pair_num = dictionary[random.sample(ingredient_pairs, 1)[0]]
        except KeyError:
            pair_num = 0
        return ingredient_num, pair_num
    except ValueError:
        # some recipes have less than two ingredients, so select a new recipe
        return get_sample_pair(data, dictionary, reverse_dictionary, ingredient_to_recipe, ingredients)

def generate_batch(batch_size, data, dictionary, reverse_dictionary,
                   ingredient_to_recipe, ingredients_train):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    for i in range(batch_size):
        batch[i], labels[i, 0] = get_sample_pair(
            data, dictionary, reverse_dictionary, ingredient_to_recipe, ingredients_train)
    return batch, labels

def plot_loss(train_loss_by_epoch, val_loss_by_epoch, filename='Loss by epoch.png'):
    plt.plot(train_loss_by_epoch)
    plt.plot(val_loss_by_epoch)
    plt.ylabel('NCE loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(config.path_outputs, filename))

def main(vocab_size=10000, verbose=1, batch_size=64, num_steps=50001, embedding_size=128):
    path_final_embedding = os.path.join(config.path_data, 'final_embeddings.pk')
    if os.path.exists(path_final_embedding):
        with open(path_final_embedding, 'rb') as f:
            final_embeddings = pickle.load(f)
        print('Embeddings loaded from disk; rm {} to refresh.'.format(path_final_embedding))
        return(final_embeddings)

    ____section____ = section_print()
    ____section____('Prep data')
    data = prep_data.main()
    ingredients_train = parse_ingredient_listlist(data.train.ingredients)
    ingredients_val = parse_ingredient_listlist(data.validation.ingredients)

    ingredients_joined_train = join_ingredients(ingredients_train)
    ingredients_joined_val = join_ingredients(ingredients_val)

    ingredients_list_train = get_flat_ingredients_list(ingredients_joined_train)
    ingredients_list_val = get_flat_ingredients_list(ingredients_joined_val)

    ____section____('Build ingredients dictionary')
    data_train, count_train, dictionary_train, reverse_dict_train = build_dataset(ingredients_list_train, vocab_size)
    data_val, count_val, dictionary_val, reverse_dict_val = build_dataset(ingredients_list_val, vocab_size)
    assert vocab_size == len(dictionary_train)
    if verbose > 1:
        print('Most common ingredients (+UNK)', count_train[:5])
        print('Sample data', data_train[:10], [reverse_dict_train[i] for i in data_train[:10]])
    ingredient_to_recipe_train = get_ingredient_to_recipe(dictionary_train.keys(), ingredients_joined_train)
    ingredient_to_recipe_val = get_ingredient_to_recipe(dictionary_val.keys(), ingredients_joined_val)

    if verbose > 1:
        ____section____('Print sample batch')
        batch, labels = generate_batch(
            16, data_train, dictionary_train, reverse_dict_train,
            ingredient_to_recipe_train, ingredients_train)
        for i in range(16):
            print(batch[i], reverse_dict_train[batch[i]], '->',
                  labels[i, 0], reverse_dict_train[labels[i, 0]])

    ____section____('Build skip-gram model')

    # Pick a random validation set to sample nearest neighbors
    valid_examples = np.random.choice(config.valid_window, config.valid_size, replace=False)

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocab_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocab_size]))

        # Compute the average NCE loss for the batch
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=config.num_sampled,
                           num_classes=vocab_size))

        # Construct the SGD optimizer
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer
        init = tf.global_variables_initializer()

    ____section____('Train skip-gram model')
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them
        init.run()
        if verbose > 0:
            print("Initialized")

        train_loss_by_epoch = []
        val_loss_by_epoch = []
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, data_train, dictionary_train, reverse_dict_train,
                ingredient_to_recipe_train, ingredients_train)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # Perform one update step by evaluating the optimizer op
            _, loss_train = session.run([optimizer, loss], feed_dict=feed_dict)

            if step % config.loss_eval_freq == 0:

                # Evaluation validation loss
                batch_inputs, batch_labels = generate_batch(
                    batch_size, data_val, dictionary_val, reverse_dict_val,
                    ingredient_to_recipe_val, ingredients_val)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

                if verbose > 0:
                    print("Average loss at step {:,} of {:,}: {:.2f} (train), {:.2f} (validation)".format(
                        step, num_steps, loss_train, loss_val))

                # Save train and validation loss for plotting
                if step > 0:
                    train_loss_by_epoch.append(loss_train)
                    val_loss_by_epoch.append(loss_val)

            # Report sample similarities
            if (verbose > 0) & (step % config.sim_eval_freq == 0):
                sim = similarity.eval()
                for i in xrange(config.valid_size):
                    valid_word = reverse_dict_train[valid_examples[i]]
                    top_k = config.num_nearest_neigh  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dict_train[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()

    ____section____('Plot results: loss developments and reduced vectors')
    plot_loss(train_loss_by_epoch, val_loss_by_epoch)
    labels = [reverse_dict_train[i] for i in xrange(config.max_plot_points)]
    plot_reduced(final_embeddings[:config.max_plot_points, :], labels)

    ____section____('Save embeddings to disk')
    embedding_dict = {l: e for l, e in zip(labels, final_embeddings)}
    with open(path_final_embedding, 'wb') as f:
        pickle.dump(embedding_dict, f)
    return embedding_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Print a sample batch prior to training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size')
    parser.add_argument('--num-steps', type=int, default=50001,
                        help='Number of training steps')
    parser.add_argument('--embedding-size', type=int, default=128,
                        help='Size of the embedded vector')
    args = parser.parse_args()
    main(args.vocab_size, args.verbose, args.batch_size, args.num_steps, args.embedding_size)

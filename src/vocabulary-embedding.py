# Generate intial word embedding for headlines and description

# The embedding is limited to a fixed vocabulary size (`vocab_size`) but
# a vocabulary of all the words that appeared in the data is built.
from os import path
import config
import _pickle as pickle
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

FN = 'vocabulary-embedding'
seed = 42
vocab_size = 40000
embedding_dim = 100
lower = False


# read tokenized headlines and descriptions
with open(path.join(config.path_data, 'tokens.pkl'), 'rb') as fp:
    heads, desc = pickle.load(fp)

if lower:
    heads = [h.lower() for h in heads]

if lower:
    desc = [h.lower() for h in desc]

# build vocabulary
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = list(map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1])))
    return vocab, vocabcount

vocab, vocabcount = get_vocab(heads + desc)

# most popular tokens
print()
print('Most popular tokens:')
print(vocab[:50])
print('Total vocab size: {:,}'.format(len(vocab)))

# plot word distribution in headlines and discription
plt.plot([vocabcount[w] for w in vocab])
plt.gca().set_xscale("log", nonposx='clip')
plt.gca().set_yscale("log", nonposy='clip')
title = 'word distribution in headlines and discription'
plt.title(title)
plt.xlabel('rank')
plt.ylabel('total appearances')
plt.savefig(path.join(config.path_outputs, '{}.png'.format(title)))

# Index words
empty = 0  # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos + 1  # first real word

def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx + start_idx) for idx, word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    idx2word = dict((idx, word) for word, idx in word2idx.items())
    return word2idx, idx2word

word2idx, idx2word = get_idx(vocab, vocabcount)

# Word Embedding
# read GloVe
glove_name = path.join(config.path_data, 'glove.6B.{}d.txt'.format(embedding_dim))

glove_n_symbols = sum(1 for line in open(glove_name))
print()
print('{:,} GloVe symbols'.format(glove_n_symbols))

glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
globale_scale = .1
with open(glove_name, 'r') as fp:
    i = 0
    for l in fp:
        l = l.strip().split()
        w = l[0]
        glove_index_dict[w] = i
        glove_embedding_weights[i, :] = list(map(float, l[1:]))
        i += 1
glove_embedding_weights *= globale_scale

print('GloVe std dev: {:.4f}'.format(glove_embedding_weights.std()))

for w, i in glove_index_dict.items():
    w = w.lower()
    if w not in glove_index_dict:
        glove_index_dict[w] = i


# ## embedding matrix

# use GloVe to initialize embedding matrix

# In[20]:

# generate random embedding with same scale as glove
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights.std() * np.sqrt(12) / 2  # uniform and not normal
embedding = np.random.uniform(low=-scale, high=scale, size=shape)
print()
print('random-embedding/glove scale: {:.4f} std: {:.4f}'.format(scale, embedding.std()))

# copy from glove weights of words that appear in our short vocabulary (idx2word)
c = 0
for i in range(vocab_size):
    w = idx2word[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is None and w.startswith('#'):  # glove has no hastags (I think...)
        w = w[1:]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is not None:
        embedding[i, :] = glove_embedding_weights[g, :]
        c += 1

print('number of tokens, in small vocab: {:,} found in glove and copied to embedding: {:.4f}'.format(c, c / float(vocab_size)))

# generate random embedding with same scale as glove
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights.std() * np.sqrt(12) / 2  # uniform and not normal
embedding = np.random.uniform(low=-scale, high=scale, size=shape)
print('random-embedding/glove scale: {:.4f} std: {:.4f}'.format(scale, embedding.std()))

# copy from glove weights of words that appear in our short vocabulary (idx2word)
c = 0
for i in range(vocab_size):
    w = idx2word[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is None and w.startswith('#'):  # glove has no hastags (I think...)
        w = w[1:]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is not None:
        embedding[i, :] = glove_embedding_weights[g, :]
        c += 1
print('number of tokens, in small vocab: {:,} found in glove and copied to embedding: {:.4f}'.format(c, c / float(vocab_size)))


# lots of word in the full vocabulary (word2idx) are outside `vocab_size`.
# Build an alterantive which will map them to their closest match in glove but only if the match
# is good enough (cos distance above `glove_thr`)

# In[24]:

glove_thr = 0.5


# In[25]:

word2glove = {}
for w in word2idx:
    if w in glove_index_dict:
        g = w
    elif w.lower() in glove_index_dict:
        g = w.lower()
    elif w.startswith('#') and w[1:] in glove_index_dict:
        g = w[1:]
    elif w.startswith('#') and w[1:].lower() in glove_index_dict:
        g = w[1:].lower()
    else:
        continue
    word2glove[w] = g


# for every word outside the embedding matrix find the closest word inside the mebedding matrix.
# Use cos distance of GloVe vectors.
# Allow for the last `nb_unknown_words` words inside the embedding matrix to be considered to be outside.
# Dont accept distances below `glove_thr`

# In[26]:

normed_embedding = embedding / np.array(
    [np.sqrt(np.dot(gweight, gweight)) for gweight in embedding])[:, None]

nb_unknown_words = 100

glove_match = []
for w, idx in word2idx.items():
    if idx >= vocab_size - nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_embedding_weights[gidx, :].copy()
        # find row in embedding that has the highest cos score with gweight
        gweight /= np.sqrt(np.dot(gweight, gweight))
        score = np.dot(normed_embedding[:vocab_size - nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_thr:
                break
            if idx2word[embedding_idx] in word2glove:
                glove_match.append((w, embedding_idx, s))
                break
            score[embedding_idx] = -1
glove_match.sort(key=lambda x: -x[2])
print()
print('# of GloVe substitutes found: {:,}'.format(len(glove_match)))


# manually check that the worst substitutions we are going to do are good enough
for orig, sub, score in glove_match[-10:]:
    print('{:.4f}'.format(score), orig, '=>', idx2word[sub])

# build a lookup table of index of outside words to index of inside words
glove_idx2idx = dict((word2idx[w], embedding_idx) for w, embedding_idx, _ in glove_match)

# Data
Y = [[word2idx[token] for token in headline.split()] for headline in heads]
plt.hist(list(map(len, Y)), bins=50)
plt.savefig(path.join(config.path_outputs, 'headline_distribtion.png'))

X = [[word2idx[token] for token in d.split()] for d in desc]
plt.hist(list(map(len, X)), bins=50)
plt.savefig(path.join(config.path_outputs, 'description_distribtion.png'))

with open(path.join(config.path_data, '{}.pkl'.format(FN)), 'wb') as fp:
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx), fp, 2)

with open(path.join(config.path_data, '{}.data.pkl'.format(FN)), 'wb') as fp:
    pickle.dump((X, Y), fp, 2)

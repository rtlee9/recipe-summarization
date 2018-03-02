"""Define constant variables."""

# define empty and end-of-sentence vocab idx
empty = 0
eos = 1

# input data (X) is made from maxlend description words followed by eos followed by
# headline words followed by eos if description is shorter than maxlend it will be
# left padded with empty if entire data is longer than maxlen it will be clipped and
# if it is shorter it will be right padded with empty. labels (Y) are the headline
# words followed by eos and clipped or padded to maxlenh. In other words the input is
# made from a maxlend half in which the description is padded from the left and a
# maxlenh half in which eos is followed by a headline followed by another eos if there
# is enough space. The labels match only the second half and the first label matches
# the eos at the start of the second half (following the description in the first half)
maxlend = 100
maxlenh = 15
maxlen = maxlend + maxlenh
activation_rnn_size = 40 if maxlend else 0
nb_unknown_words = 10

# function names
FN0 = 'vocabulary-embedding'  # filename of vocab embeddings
FN1 = 'train'  # filename of model weights

# training variables
seed = 42
optimizer = 'adam'
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
regularizer = None

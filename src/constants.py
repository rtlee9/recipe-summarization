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

"""Generate samples.

Variation on https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
"""
import Levenshtein
import numpy as np
import random
from keras.preprocessing import sequence

from constants import empty, eos, maxlend, maxlenh, maxlen


def lpadd(x):
    """Left (pre) pad a description to maxlend and then add eos.

    The eos is the input to predicting the first word in the headline
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty] * (maxlend - n) + x + [eos]


def beamsearch(
        predict, start, k, maxsample, use_unk, empty, temperature, nb_unknown_words,
        vocab_size, model, batch_size, avoid=None, avoid_score=1):
    """Return k samples (beams) and their NLL scores, each sample is a sequence of labels.

    All samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples.
    """
    def sample(energy, n, temperature=temperature):
        """Sample at most n elements according to their energy."""
        n = min(n, len(energy))
        prb = np.exp(-np.array(energy) / temperature)
        res = []
        for i in range(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb / z, 1))
            res.append(r)
            prb[r] = 0.  # make sure we select each element only once
        return res

    dead_samples = []
    dead_scores = []
    live_k = 1  # samples that did not yet reached eos
    live_samples = [list(start)]
    live_scores = [0]

    while live_k:
        # for every possible live sample calc prob for every possible label
        probs = predict(live_samples, empty=empty, model=model, batch_size=batch_size)

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:, None] - np.log(probs)
        cand_scores[:, empty] = 1e20
        if not use_unk:
            for i in range(nb_unknown_words):
                cand_scores[:, vocab_size - 1 - i] = 1e20

        if avoid:
            for a in avoid:
                for i, s in enumerate(live_samples):
                    n = len(s) - len(start)
                    if n < len(a):
                        # at this point live_sample is before the new word,
                        # which should be avoided, is added
                        cand_scores[i, a[n]] += avoid_score

        live_scores = list(cand_scores.flatten())

        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)
        ranks_dead = [r for r in ranks if r < n]
        ranks_live = [r - n for r in ranks if r >= n]

        dead_scores = [dead_scores[r] for r in ranks_dead]
        dead_samples = [dead_samples[r] for r in ranks_dead]

        live_scores = [live_scores[r] for r in ranks_live]

        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        live_samples = [live_samples[r // voc_size] + [r % voc_size] for r in ranks_live]

        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of maxlenh
        zombie = [s[-1] == eos or len(s) > maxsample for s in live_samples]

        # add zombies to the dead
        dead_samples += [s for s, z in zip(live_samples, zombie) if z]
        dead_scores += [s for s, z in zip(live_scores, zombie) if z]
        # remove zombies from the living
        live_samples = [s for s, z in zip(live_samples, zombie) if not z]
        live_scores = [s for s, z in zip(live_scores, zombie) if not z]
        live_k = len(live_samples)

    return dead_samples + live_samples, dead_scores + live_scores


def keras_rnn_predict(samples, empty, model, batch_size):
    """For every sample, calculate probability for every possible label.

    You need to supply your RNN model and maxlen - the length of sequences it can handle
    """
    sample_lengths = list(map(len, samples))
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    # pad from right (post) so the first maxlend will be description followed by headline
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    return np.array([prob[sample_length - maxlend - 1]
                     for prob, sample_length in zip(probs, sample_lengths)])


def vocab_fold(xs, oov0, glove_idx2idx, vocab_size, nb_unknown_words):
    """Convert list of word indices that may contain words outside vocab_size to words inside.

    If a word is outside, try first to use glove_idx2idx to find a similar word inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    xs = [x if x < oov0 else glove_idx2idx.get(x, x) for x in xs]
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= oov0])
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x, vocab_size - 1 - min(i, nb_unknown_words - 1)) for i, x in enumerate(outside))
    xs = [outside.get(x, x) for x in xs]
    return xs


def vocab_unfold(desc, xs, oov0):
    """Covert a description to a list of word indices."""
    # assume desc is the unfolded version of the start of xs
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= oov0:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x, x) for x in xs]


def gensamples(
        skips, short, data, idx2word, oov0, glove_idx2idx, vocab_size,
        nb_unknown_words, avoid=None, avoid_score=1, **kwargs):
    """Generate text samples."""
    # unpack data
    X, Y = data

    # if data is full dataset pick a random header and description
    if not isinstance(X[0], int):
        i = random.randint(0, len(X) - 1)
        x = X[i]
        y = Y[i]
    else:
        x = X
        y = Y

    # print header and description
    print('HEAD:', ' '.join(idx2word[w] for w in y[:maxlenh]))
    print('DESC:', ' '.join(idx2word[w] for w in x[:maxlend]))

    if avoid:
        # avoid is a list of avoids. Each avoid is a string or list of word indeicies
        if isinstance(avoid, str) or isinstance(avoid[0], int):
            avoid[avoid]
        avoid = [a.split() if isinstance(a, str) else a for a in avoid]
        avoid = [[a] for a in avoid]

    print('HEADS:')
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend, len(x)), max(maxlend, len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start, oov0, glove_idx2idx, vocab_size, nb_unknown_words)
        sample, score = beamsearch(
            predict=keras_rnn_predict,
            start=fold_start,
            maxsample=maxlen,
            empty=empty,
            nb_unknown_words=nb_unknown_words,
            vocab_size=vocab_size,
            avoid=avoid,
            **kwargs
        )
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s, start, scr) for s, scr in zip(sample, score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample, oov0)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w // (256 * 256)) + chr((w // 256) % 256) + chr(w % 256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code, c) for c in codes])
            if distance > -0.6:
                print(score, ' '.join(words))
        else:
                print(score, ' '.join(words))
        codes.append(code)
    return samples

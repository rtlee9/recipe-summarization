"""Utility methods."""


def join_ingredients(ingredients_listlist):
    """Join multiple lists of ingredients with ' , '."""
    return [' , '.join(i) for i in ingredients_listlist]


def get_flat_ingredients_list(ingredients_joined_train):
    """Flatten lists of ingredients encoded as a string into a single list."""
    return ' , '.join(ingredients_joined_train).split(' , ')


def section_print():
    """Memorized function keeping track of section number."""
    section_number = 0

    def inner(message):
        """Print section number."""
        global section_number
        section_number += 1
        print('Section {}: {}'.format(section_number, message))
    print('Section {}: initializing section function'.format(section_number))
    return inner


def is_filename_char(x):
    """Return True if x is an acceptable filename character."""
    if x.isalnum():
        return True
    if x in ['-', '_']:
        return True
    return False


def url_to_filename(filename):
    """Map a URL string to filename by removing unacceptable characters."""
    return "".join(x for x in filename if is_filename_char(x))


def prt(label, word_idx, idx2word):
    """Map `word_idx` list to words and print it with its associated `label`."""
    words = [idx2word[word] for word in word_idx]
    print('{}: {}\n'.format(label, ' '.join(words)))


def str_shape(x):
    """Format the dimension of numpy array `x` as a string."""
    return 'x'.join([str(element) for element in x.shape])

if __name__ == '__main__':
    print(url_to_filename('http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename'))

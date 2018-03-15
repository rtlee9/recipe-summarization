"""Tokenize recipes."""
import _pickle as pickle
from os import path
from nltk.tokenize import word_tokenize
from nltk import download
from tqdm import tqdm

import config
import prep_data
from parse_ingredients import parse_ingredient_list


def tokenize_sentence(sentence):
    """Tokenize a sentence."""
    try:
        return ' '.join(list(filter(
            lambda x: x.lower() != "advertisement",
            word_tokenize(sentence))))
    except LookupError:
        print('Downloading NLTK data')
        download()
        return ' '.join(list(filter(
            lambda x: x.lower() != "advertisement",
            word_tokenize(sentence))))


def recipe_is_complete(r):
    """Return True if recipe is complete and False otherwise.

    Completeness is defined as the recipe containing a title and instructions.
    """
    if ('title' not in r) or ('instructions' not in r):
        return False
    if (r['title'] is None) or (r['instructions'] is None):
        return False
    return True


def tokenize_recipes(recipes):
    """Tokenise all recipes."""
    tokenized = []
    for r in tqdm(recipes.values()):
        if recipe_is_complete(r):
            ingredients = '; '.join(parse_ingredient_list(r['ingredients'])) + '; '
            tokenized.append((
                tokenize_sentence(r['title']),
                tokenize_sentence(ingredients) + tokenize_sentence(r['instructions'])))
    return tuple(map(list, zip(*tokenized)))


def pickle_recipes(recipes):
    """Pickle all recipe tokens to disk."""
    with open(path.join(config.path_data, 'tokens.pkl'), 'wb') as f:
        pickle.dump(recipes, f, 2)


def load_recipes():
    """Read pickled recipe tokens from disk."""
    with open(path.join(config.path_data, 'tokens.pkl'), 'rb') as f:
        recipes = pickle.load(f)
    return recipes


def main():
    """Tokenize recipes."""
    recipes = prep_data.load_recipes()
    text_sum_data = tokenize_recipes(recipes)
    pickle_recipes(text_sum_data)

if __name__ == '__main__':
    main()

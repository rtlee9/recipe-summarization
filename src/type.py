"""Define named tuple types."""
from collections import namedtuple

RecipeContainer = namedtuple('RecipeContainer', ['keys', 'titles', 'ingredients', 'directions', 'images'])
GANContainer = namedtuple('GANContainer', ['images', 'embeddings', 'recipe_names'])
DataContainer = namedtuple('DataContainer', ['train', 'validation', 'test'])

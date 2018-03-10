"""Preprocess recipe data for training."""
import os
from os import path
from glob import glob
import json
import re
import pickle
import argparse

import numpy as np
from textwrap import wrap
from scipy import ndimage, misc

import config
from type import RecipeContainer, DataContainer
from utils import url_to_filename

# import matplotlib using agg backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def get_train_val_test_keys(keys, val_pct=.1, test_pct=.1):
    """Split a list of keys into three groups: train, validation, and test."""
    n = keys.shape[0]
    np.random.shuffle(keys)
    test_cutoff = 1 - test_pct
    val_cutoff = test_cutoff - val_pct
    return np.split(keys, [int(val_cutoff * n), int(test_cutoff * n)])


def files_to_containers(files, recipes, image_list):
    """Store recipe components as a data container."""
    images = np.array([image_list[f] for f in files])
    titles = np.array([recipes[f]['title'] for f in files])
    ingredients = np.array([recipes[f]['ingredients'] for f in files])
    directions = np.array([recipes[f]['instructions'] for f in files])
    return RecipeContainer(files, titles, ingredients, directions, images)


def get_plt_grid(df, labels, subplot_shape=(4, 6), fig_size=(12, 8)):
    """Return a matplotlib grid of randomly selected images from dataframe df of shape subplot_shape."""
    fig, axes = plt.subplots(*subplot_shape)
    for ax in axes.ravel():
        rand_index = np.random.randint(0, df.shape[0])
        img = df[rand_index]
        label = labels[rand_index]
        ax.imshow(img)
        ax.set_title(
            '\n'.join(wrap(label, 25)), y=.95, va='top', size=8,
            bbox=dict(facecolor='white', pad=.1, alpha=0.6, edgecolor='none'))
        ax.axis('off')
    fig.tight_layout()
    fig.set_size_inches(*fig_size)
    fig.subplots_adjust(wspace=.0, hspace=.0)
    return fig


def load_recipe(filename):
    """Load a single recipe collection from disk."""
    with open(filename, 'r') as f:
        recipes = json.load(f)
    print('Loaded {:,} recipes from {}'.format(len(recipes), filename))
    return recipes


def clean_recipe_keys(recipes):
    """Clean recipe keys by stripping URLs of special characters."""
    recipes_clean = {}
    for key, value in recipes.items():
        recipes_clean[url_to_filename(key)] = value
    return recipes_clean


def load_recipes():
    """Load all recipe collections from disk and combine into single dataset."""
    recipes = {}
    for filename in glob(path.join(config.path_recipe_box_data, 'recipes_raw*.json')):
        recipes.update(load_recipe(filename))
    print('Loaded {:,} recipes in total'.format(len(recipes)))
    return clean_recipe_keys(recipes)


def load_images(img_dims):
    """Load all images into a dictionary with filename as the key and numpy image array as the value."""
    image_list = {}
    for root, dirnames, filenames in os.walk(config.path_img):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                filepath = os.path.join(root, filename)
                try:
                    image = ndimage.imread(filepath, mode="RGB")
                except OSError:
                    print('Could not load image {}'.format(filepath))
                image_resized = misc.imresize(image, img_dims)
                if np.random.random() > 0.5:
                    # Flip horizontally with probability 50%
                    image_resized = np.fliplr(image_resized)
                image_list[filename.split('.')[0]] = image_resized
    print('Loaded {:,} images from disk'.format(len(image_list)))
    return image_list


def _get_shape_str(shape):
    """Convert image shape to string for filename description."""
    return '{}_{}'.format(*shape[:2])


def _get_npy_filename(shape):
    """Return absolute path for npy image file."""
    shape_str = _get_shape_str(shape)
    return path.join(
        config.path_recipe_box_data, 'images_processed_{}.npy'.format(shape_str))


def _get_filename_filename(shape):
    """Return absolute path for filename storage."""
    shape_str = _get_shape_str(shape)
    return path.join(
        config.path_recipe_box_data, 'images_processed_filenames_{}.pk'.format(shape_str))


def save_images(image_list):
    """Save images and associated keys to disk."""
    filenames, images = list(image_list.keys()), np.array(list(image_list.values()))
    shape = images.shape[1:]
    np.save(_get_npy_filename(shape), images)
    with open(_get_filename_filename(shape), 'wb') as f:
        pickle.dump(filenames, f)
    print('Saved {:,} images to disk'.format(images.shape[0]))


def load_images_disk(shape):
    """Load preprocessed images and associated keys from disk."""
    images = np.load(_get_npy_filename(shape))
    with open(_get_filename_filename(shape), 'rb') as f:
        filenames = pickle.load(f)
    print('Loaded {:,} preprocessed images from disk'.format(images.shape[0]))
    return {f: i for f, i in zip(filenames, images)}


def smart_load_images(img_dims):
    """Load preprocessed images and associated keys from disk if available.

    Otherwise, load raw images from disk, process, then save to disk.
    """
    path_load = _get_npy_filename(img_dims)
    if path.exists(path_load):
        return(load_images_disk(img_dims))
    else:
        images = load_images(img_dims)
        save_images(images)
        return(images)


def plot_grids_by_segment(data):
    """Plot image sample."""
    get_plt_grid(data.train.images, data.train.titles).savefig(
        path.join(config.path_outputs, 'sample-train-imgs.png'))
    get_plt_grid(data.validation.images, data.validation.titles).savefig(
        path.join(config.path_outputs, 'sample-validation-imgs.png'))
    get_plt_grid(data.test.images, data.test.titles).savefig(
        path.join(config.path_outputs, 'sample-test-imgs.png'))


def get_complete_recipes(recipes, image_list):
    """Return intersection of recipe keys and image keys."""
    recipe_keys = [url_to_filename(k) for k in recipes.keys()]
    files = np.array([filename for filename in image_list.keys()
                      if filename in recipe_keys])
    print('{:,} complete recipes found'.format(len(files)))
    return files


def save_data_container(data, filename_pickle):
    """Save data container to disk in multiple pieces to keep under 2GB limit."""
    with open(filename_pickle + '_train.pk', 'wb') as f:
        pickle.dump(data.train, f)
    with open(filename_pickle + '_validation.pk', 'wb') as f:
        pickle.dump(data.validation, f)
    with open(filename_pickle + '_test.pk', 'wb') as f:
        pickle.dump(data.test, f)
    print('Data container saved to {}_*.pk'.format(filename_pickle))


def load_recipe_container(filename_pickle):
    """Load data containers from disk and create super container."""
    with open(filename_pickle + '_train.pk', 'rb') as f:
        train = pickle.load(f)
    with open(filename_pickle + '_validation.pk', 'rb') as f:
        validation = pickle.load(f)
    with open(filename_pickle + '_test.pk', 'rb') as f:
        test = pickle.load(f)
    return DataContainer(train, validation, test)


def save_recipes(filename_pickle, batch_size):
    """Load preprocessed data container from disk if available.

    Otherwise, create container and then save to disk
    """
    # Load recipes and images
    recipes = load_recipes()
    image_list = smart_load_images(batch_size)

    # Get train, validation, test split
    files = get_complete_recipes(recipes, image_list)
    train_files, validation_files, test_files = get_train_val_test_keys(files)
    print('Data split into segments of size {:,} (train), {:,} (validation), and {:,} (test)'.format(
        train_files.shape[0], validation_files.shape[0], test_files.shape[0]))

    # Save data in container
    data = DataContainer(
        files_to_containers(train_files, recipes, image_list),
        files_to_containers(validation_files, recipes, image_list),
        files_to_containers(test_files, recipes, image_list),
    )
    save_data_container(data, filename_pickle)

    # Plot image sample
    plot_grids_by_segment(data)

    return data


def pickled_data_container_exists(filename_pickle):
    """Check whether pickled data container exists at expected path."""
    if not path.exists(filename_pickle + '_train.pk'):
        return False
    elif not path.exists(filename_pickle + '_validation.pk'):
        return False
    elif not path.exists(filename_pickle + '_test.pk'):
        return False
    else:
        return True


def main(img_size=64):
    """Return a single data container containing all recipe components, split into train, validation, and test sets."""
    filename_pickle = path.join(config.path_data, 'data_processed')
    if not pickled_data_container_exists(filename_pickle):
        data = save_recipes(filename_pickle, (img_size, img_size))
    else:
        print('Loading pickled data; rm {} to refresh'.format(filename_pickle))
        data = load_recipe_container(filename_pickle)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
    opt = parser.parse_args()
    main(opt.batch_size)

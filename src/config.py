from os import path, makedirs

path_src = path.dirname(path.abspath(__file__))
path_base = path.dirname(path_src)
path_scrapers = path.join(path_base, 'recipe-box')
path_recipe_box_data = path.join(path_scrapers, 'data')
path_data = path.join(path_base, 'data')
path_models = path.join(path_base, 'models')
path_logs = path.join(path_models, 'logs')
path_img = path.join(path_recipe_box_data, 'img')
path_outputs = path.join(path_base, 'outputs')

# verify output path exists otherwise make it
for p in [path_data, path_outputs, path_models, path_logs, path_img]:
    if not path.exists(p):
        makedirs(p)

# ****************************************** #
# Training config
# ****************************************** #
valid_size = 16     # Random set of ingredients to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
num_sampled = 64    # Number of negative examples to sample.

loss_eval_freq = 2000
sim_eval_freq = 50000
num_nearest_neigh = 8

max_plot_points = 500
tsne_iter = 5000
tsne_perplexity = 30

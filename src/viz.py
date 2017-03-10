from os import path
from sklearn.manifold import TSNE

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from textwrap import wrap

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.embed import file_html
from bokeh.resources import CDN

import config


def plot_embeddings(vecs, labels, plot_desc="ingredient representations",
                    dot_size=5, alpha=.5, height=500, width=500):
    source = ColumnDataSource(
        data=dict(
            x=vecs[:, 0].tolist(),
            y=vecs[:, 1].tolist(),
            name=labels,
        )
    )
    hover = HoverTool(
        tooltips=[
            ("Name", "@name"),
        ]
    )
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    plot = figure(tools=[hover, TOOLS], active_scroll='wheel_zoom', width=width, height=height)
    plot.scatter(
        'x', 'y', source=source, alpha=.5, size=dot_size)
    plot.toolbar.logo = None
    plot.axis.visible = False
    plot.grid.visible = False
    html = file_html(plot, CDN, plot_desc)
    with open(path.join(config.path_outputs, '{}.html'.format(plot_desc)), 'w') as f:
        f.write(html)

def plot_reduced(final_embeddings, labels):
    """Reduce dimensionality with t-SNE, plot and save
    """
    tsne = TSNE(
        perplexity=config.tsne_perplexity,
        n_components=2,
        init='pca',
        n_iter=config.tsne_iter,
    )
    low_dim_embs = tsne.fit_transform(final_embeddings)
    plot_embeddings(low_dim_embs, labels)

def get_full_plt_grid(df, labels, fig_size=(12, 12)):
    """Return a matplotlib grid of all images
    from dataframe df
    """
    N = df.shape[0]
    subplot_shape = np.tile(np.int(np.sqrt(N)), 2)
    fig, axes = plt.subplots(*subplot_shape)
    for i, ax in enumerate(axes.ravel()):
        img = df[i]
        label = labels[i]
        ax.imshow(img)
        ax.set_title(
            '\n'.join(wrap(label, 25)), y=.95, va='top', size=8,
            bbox=dict(facecolor='white', pad=.1, alpha=0.6, edgecolor='none'))
        ax.axis('off')
    fig.tight_layout()
    fig.set_size_inches(*fig_size)
    fig.subplots_adjust(wspace=.0, hspace=.0)
    return fig

def plot_tensor_grid(imgs_tensor, recipe_names, filepath):
    imgs_tensor_norm = (imgs_tensor + 1) / 2 * 255
    imgs_numpy = imgs_tensor_norm.cpu().numpy().astype('uint8').transpose(0, 2, 3, 1)
    plt = get_full_plt_grid(imgs_numpy, recipe_names, (13, 12))
    plt.savefig(filepath)

import pandas as pd
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from os.path import join
from seaborn import clustermap
from scipy.stats import spearmanr


def main():
    orn_responses = pd.read_csv('HC_data_raw.csv', header=[0, 1], index_col=[0, 1])
    # spontaneous = orn_responses.loc[(0, 'spontaneous firing rate')].to_numpy()
    # df = orn_responses + spontaneous
    # df[df < 0] = 0
    df = orn_responses
    df = df.drop(index=(0, 'spontaneous firing rate'))

    community = {'dm2', 'dp1m', 'vm2', 'dl2v', 'dm3', 'dm4', 'dm1', 'vm3', 'va2', 'va4'}
    lut = {}
    for i in df.columns:
        if i[0] in community:
            lut[i] = 'g'
        else:
            lut[i] = 'k'
    row_colors = df.columns.map(lut)
    fruits = df.loc[12, :].T
    molecules = df.droplevel(0).drop(fruits.columns).T
    df = df.T

    # fruits = fruits.loc[:, ~fruits.columns.str.contains('pure')]
    # fruits = fruits.loc[:, ~fruits.columns.str.contains('-2')]

    spearman = lambda u, v: 1 - spearmanr(u, v)[0]
    metric = spearman
    metric_name = 'spearman' if metric == spearman else metric

    gf = clustermap(fruits, metric=metric, row_colors=row_colors)
    gf.fig.suptitle(metric_name + ' fruits')
    gf.savefig(metric_name + '_clustermap_fruits.png')

    gm = clustermap(molecules, metric=metric, col_cluster=False, row_colors=row_colors)
    gm.fig.suptitle(metric_name + ' monomolecular')
    gm.savefig(metric_name + '_clustermap_monomolecular.png')

    gdf = clustermap(df, metric=metric, col_cluster=False, row_colors=row_colors)
    gdf.fig.suptitle(metric_name + ' df')
    gdf.savefig(metric_name + '_clustermap_df.png')

    Y_fruits = pdist(fruits, metric)
    Y_molecules = pdist(molecules, metric)

    plt.figure()
    plt.hist(Y_fruits, bins='auto')
    plt.title(metric_name + ' fruits')
    plt.savefig(metric_name + '_fruits.png')

    plt.figure()
    plt.hist(Y_molecules, bins='auto')
    plt.title(metric_name + ' monomolecular')
    plt.savefig(metric_name + '_monomolecular.png')


if __name__ == '__main__':
    main()

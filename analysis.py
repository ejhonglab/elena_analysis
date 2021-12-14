from os.path import join, exists
from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import cv2

from hong2p import util, thor


def main():
    stimfile_dir = r'E:\research\ejhonglab\2021-11-30\4'
    thorimage_dir = join(stimfile_dir, 'flyfood')
    thorsync_dir = join(stimfile_dir, 'SyncData')
    ignore_bounding_frame_cache = False
    generate_csv = False    # quantify intensity

    # options for correlation matrix:
    sort_corr_mat = True    # group correlation matrix by odor type instead of presentation order
    filter_movie = False    # apply a 5x5 pixel gaussian kernel to filter the movie
    thresh_movie = False    # apply cv2 THRESH_TOZERO to max pixel values

    yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(thorimage_dir, stimfile_dir)
    odors = {
        'control': {i['name'] for i in yaml_data['pins2odors'].values() if i['type'] == 'control'},
        'flyfood': {i['name'] for i in yaml_data['pins2odors'].values() if i['type'] == 'flyfood'}
    }

    bounding_frame_yaml_cache = join(stimfile_dir, 'trial_bounding_frames.yaml')
    if ignore_bounding_frame_cache or not exists(bounding_frame_yaml_cache):
        bounding_frames = thor.assign_frames_to_odor_presentations(thorsync_dir, thorimage_dir)
        assert len(bounding_frames) == len(odor_lists)
        bounding_frames = [[int(x) for x in xs] for xs in bounding_frames]

        with open(bounding_frame_yaml_cache, 'w') as f:
            yaml.dump(bounding_frames, f)
    else:
        with open(bounding_frame_yaml_cache, 'r') as f:
            bounding_frames = yaml.safe_load(f)

    movie = thor.read_movie(thorimage_dir)
    masks = util.ijroi_masks(thorimage_dir, thorimage_dir)
    merged = util.merge_ijroi_masks(masks)
    traces = util.extract_traces_bool_masks(movie, merged)

    if filter_movie:
        movie = filtered(movie)
        fig_name = 'smooth_correlation.png'
    else:
        fig_name = 'correlation.png'

    # df/f plot
    fig, axs = plt.subplots(traces.shape[1], sharex='col', figsize=(16, 6))
    cmap = {'#f6e8c3': '1 flyfood', '#d8b365': '4 flyfood', '#e08e2b': '5 flyfood',
            '#c7eae5': '1 control', '#5ab4ac': '4 control', '#28ada2': '5 control'}
    dff_full = np.zeros(shape=traces.shape)

    # csv
    single_odor_lists = []
    multi_odor_lists = []

    # correlation matrix
    dff_full_trial = {}
    dff_full_trial_legend = []
    merged = merged.to_numpy().reshape((5, 192, 192))

    response_volumes = 3

    for presentation_index in range(len(bounding_frames)):
        start_frame, first_odor_frame, end_frame = bounding_frames[presentation_index]
        baseline = traces[start_frame:(first_odor_frame - 1)].mean(axis=0)
        dff = (traces[first_odor_frame:(end_frame + 1)] - baseline) / baseline
        dff_full[start_frame:(end_frame + 1)] = (traces[start_frame:(end_frame + 1)] - baseline) / baseline

        # 1-d array for correlation matrix
        movie_baseline = movie[start_frame:(first_odor_frame - 1)].mean(axis=0)
        movie_dff = (movie[first_odor_frame:first_odor_frame + response_volumes] - movie_baseline) / movie_baseline
        movie_dff_max = np.amax(movie_dff, axis=0)
        max_dff = np.amax(dff[:response_volumes])

        # csv
        if len(odor_lists[presentation_index]) == 1:
            temp = odor_lists[presentation_index][0].copy()
            temp['trial_max_dffs'] = max_dff
            single_odor_lists.append(temp)

            color = list(cmap.keys())[0] if temp['type'] == 'flyfood' else list(cmap.keys())[3]
            axs.axvspan(first_odor_frame, end_frame, color=color, label=cmap[color])
        else:
            temp = {
                'type': odor_lists[presentation_index][0]['type'],
                'trial_max_dffs': max_dff
            }
            if len(odor_lists[presentation_index]) == 5:
                temp['name'] = temp['type']

                color = list(cmap.keys())[2] if temp['type'] == 'flyfood' else list(cmap.keys())[5]
                axs.axvspan(first_odor_frame, end_frame, color=color, label=cmap[color])
            else:
                diff = odors[temp['type']].difference({i['name'] for i in odor_lists[presentation_index]})
                temp['name'] = '-' + diff.pop()

                color = list(cmap.keys())[1] if temp['type'] == 'flyfood' else list(cmap.keys())[4]
                axs.axvspan(first_odor_frame, end_frame, color=color, label=cmap[color])

            multi_odor_lists.append(temp)
        dff_full_trial_legend.append(temp['name'])
        dff_full_trial[temp['name'] + str(presentation_index % 3)] = movie_dff_max[merged]

    # print presentation order and save to pickle file
    odor_list_unique_cache = join(stimfile_dir, 'odor_list_unqiue.p')
    print(list(OrderedDict.fromkeys(dff_full_trial_legend)))
    pickle.dump(list(OrderedDict.fromkeys(dff_full_trial_legend)), open(odor_list_unique_cache, "wb"))

    # plot df/f
    [axs.plot(dff_full[:, i], color='k') for i in range(dff_full.shape[1])]
    handles, labels = axs.get_legend_handles_labels()
    ax_label = dict(zip(labels, handles))

    axs.legend(ax_label.values(), ax_label.keys())
    file_name = 'merged_dff.svg'
    plt.savefig(join(stimfile_dir, file_name))

    if generate_csv:
        pd.DataFrame(single_odor_lists).to_csv(join(stimfile_dir, 'single_trial_max_dffs.csv'))
        pd.DataFrame(multi_odor_lists).to_csv(join(stimfile_dir, 'multi_trial_max_dffs.csv'))

    # plot correlation
    plot_correlation(dff_full_trial, stimfile_dir, thresh_movie, sort_corr_mat, len(bounding_frames), dff_full_trial_legend, fig_name)


def get_order():
    """group odors by type.
    output:
        odors: odors grouped by type with 3 repeats
        odors_ind: same as 'odors' but each repeat is labeled 0, 1, or 2
    """
    yaml_file = r'D:\research\ejhonglab\elena_olfactometer_configs\test.yaml'
    with open(yaml_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    odors_flyfood = [i['name'] for i in data_loaded['odors'] if i['type'] == 'flyfood'] + ['-'+i['name'] for i in data_loaded['odors'] if i['type'] == 'flyfood'] + ['flyfood']
    odors_control = [i['name'] for i in data_loaded['odors'] if i['type'] == 'control'] + ['-'+i['name'] for i in data_loaded['odors'] if i['type'] == 'control'] + ['control']
    odors = np.repeat(odors_flyfood + odors_control, 3)
    odors_ind = [odor + str(i % 3) for (i, odor) in enumerate(odors)]

    return odors_ind, odors


def filtered(movie):
    """apply a 5x5 pixel gaussian kernel to filter the movie. """
    for i in range(movie.shape[0]):
        for j in range(movie.shape[1]):
            movie[i, j, :, :] = cv2.GaussianBlur(movie[i, j, :, :], (5, 5), 0)
    return movie


def plot_correlation(dff_full_trial, stimfile_dir, thresh_movie, sort_corr_mat, nn, dff_full_trial_legend, fig_name):
    dff_full_trial_df = pd.DataFrame(dff_full_trial)
    if thresh_movie:
        thresh = np.percentile(dff_full_trial_df, 50)
        max_value = np.amax(dff_full_trial_df.to_numpy())
        dff_full_trial_df.loc[:, :] = cv2.threshold(dff_full_trial_df.to_numpy(), thresh, max_value, cv2.THRESH_TOZERO)[1]
        fig_name = 'thresh_' + fig_name

    if sort_corr_mat:
        odors_ind, odors = get_order()
        dff_full_trial_df = dff_full_trial_df.loc[:, odors_ind]
        dff_full_trial_legend = odors
        fig_name = 'sorted_' + fig_name

    corr_mat = dff_full_trial_df.corr()

    fig, ax = plt.subplots()
    im = ax.imshow(corr_mat, vmax=np.amax(np.triu(corr_mat, 1)))

    ax.set_xticks(np.arange(nn))
    ax.set_xticklabels(dff_full_trial_legend, rotation=90)
    ax.set_xticks(np.arange(1, nn, 3))

    ax.set_yticks(np.arange(nn))
    ax.set_yticklabels(dff_full_trial_legend)
    ax.set_yticks(np.arange(1, nn, 3))

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(join(stimfile_dir, fig_name))


if __name__ == '__main__':
    main()

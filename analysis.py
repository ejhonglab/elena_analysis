from os.path import join, exists

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from hong2p import util, thor


def main():
    stimfile_dir = r'E:\research\ejhonglab\2021-11-17\2'
    thorimage_dir = join(stimfile_dir, 'flyfood')
    thorsync_dir = join(stimfile_dir, 'SyncData001')
    ignore_bounding_frame_cache = False
    generate_csv = False # csv files for quantifying intensity

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

    fig, axs = plt.subplots(traces.shape[1], sharex='col', figsize=(16, 6)) # df/f plot
    cmap = {'#f6e8c3': '1 flyfood', '#d8b365': '4 flyfood', '#e08e2b': '5 flyfood',
            '#c7eae5': '1 control', '#5ab4ac': '4 control', '#28ada2': '5 control'}

    # for csv
    single_odor_lists = []
    multi_odor_lists = []

    # for df/f
    dff_full = np.zeros(shape=traces.shape)

    # for correlation matrix
    dff_full_trial = []
    dff_full_trial_legend = []
    merged = merged.to_numpy().reshape((1, 5, 192, 192))

    for presentation_index in range(len(bounding_frames)):
        start_frame, first_odor_frame, end_frame = bounding_frames[presentation_index]
        baseline = traces[start_frame:(first_odor_frame - 1)].mean(axis=0)
        dff = (traces[first_odor_frame:(end_frame + 1)] - baseline) / baseline
        dff_full[start_frame:(end_frame + 1)] = (traces[start_frame:(end_frame + 1)] - baseline) / baseline

        # 1-d array for correlation
        movie_baseline = movie[start_frame:(first_odor_frame - 1)].mean(axis=0)
        movie_dff = (movie[first_odor_frame:(end_frame + 1)] - movie_baseline) / movie_baseline
        # dff_full_trial.append(movie_dff.flatten())
        merged_trial = np.broadcast_to(merged, movie_dff.shape)
        dff_full_trial.append(movie_dff[merged_trial]) # pixels in ROI

        response_volumes = 3
        max_dff = np.amax(dff[:response_volumes]) # changed from mean to max

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

    # plot df/f
    [axs.plot(dff_full[:, i], color='k') for i in range(dff_full.shape[1])]
    handles, labels = axs.get_legend_handles_labels()
    ax_label = dict(zip(labels, handles))

    axs.legend(ax_label.values(), ax_label.keys())
    # axs.legend(ax_label.values(), ax_label.keys(), bbox_to_anchor=(0, 1.5, 1, 0.2), loc="upper left", mode="expand", ncol=6)
    file_name = 'merged_dff.svg'
    plt.savefig(join(stimfile_dir, file_name))

    if generate_csv:
        pd.DataFrame(single_odor_lists).to_csv(join(stimfile_dir, 'single_trial_max_dffs.csv'))
        pd.DataFrame(multi_odor_lists).to_csv(join(stimfile_dir, 'multi_trial_max_dffs.csv'))

    # plot correlation
    dff_full_trial_df = pd.DataFrame(dff_full_trial).T
    corr_mat = dff_full_trial_df.corr()
    np.fill_diagonal(corr_mat.values, 0)

    fig, ax = plt.subplots()
    im = ax.imshow(corr_mat)

    ax.set_xticks(np.arange(len(odor_lists)))
    ax.set_xticklabels(dff_full_trial_legend, rotation=90)
    ax.set_xticks(np.arange(1, len(odor_lists), 3))

    ax.set_yticks(np.arange(len(odor_lists)))
    ax.set_yticklabels(dff_full_trial_legend)
    ax.set_yticks(np.arange(1, len(odor_lists), 3))

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(join(stimfile_dir, 'correlation.png'))


if __name__ == '__main__':
    main()

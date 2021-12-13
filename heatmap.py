import os

import numpy as np
import yaml
import matplotlib.pyplot as plt

from hong2p import util, thor


def main():
    stimfile_dir = r'E:\research\ejhonglab\2021-11-30\3'
    os.mkdir(os.path.join(stimfile_dir, 'heatmap'))
    thorimage_dir = os.path.join(stimfile_dir, 'flyfood')
    bounding_frame_yaml_cache = os.path.join(stimfile_dir, 'trial_bounding_frames.yaml')
    movie = thor.read_movie(thorimage_dir)

    yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(thorimage_dir, stimfile_dir)

    with open(bounding_frame_yaml_cache, 'r') as f:
        bounding_frames = yaml.safe_load(f)

    response_volumes = 3
    for presentation_index in range(len(bounding_frames) // 3):
        fig, ax = plt.subplots(nrows=3, ncols=5)
        for repeat_index in range(3):
            start_frame, first_odor_frame, end_frame = bounding_frames[presentation_index * 3 + repeat_index]
            movie_baseline = movie[start_frame:(first_odor_frame - 1)].mean(axis=0)
            movie_dff = (movie[first_odor_frame:first_odor_frame + response_volumes] - movie_baseline) / movie_baseline
            movie_dff_max = np.amax(movie_dff, axis=0)

            for z in range(5):
                ax[repeat_index, z].imshow(movie_dff_max[z, :, :])
                ax[repeat_index, z].axis("off")

        odor_name = [i['name'] for i in odor_lists[presentation_index * 3 + repeat_index]]
        fig.suptitle(odor_name, wrap=True)
        fig.savefig(os.path.join(stimfile_dir, 'heatmap', str(presentation_index) + '_heatmap.png'))
        plt.clf()
        plt.close('all')


if __name__ == '__main__':
    main()

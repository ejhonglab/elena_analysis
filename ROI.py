import os
import pickle

import numpy as np
import pandas as pd
import yaml

from hong2p import util, thor
from analysis import filtered, plot_correlation


def main():
    write_dff_movie = False  # save the df/f movie to .tif
    # options for correlation matrix:
    sort_corr_mat = True    # group correlation matrix by odor type instead of presentation order
    filter_movie = False    # apply a 5x5 pixel gaussian kernel to filter the movie
    thresh_movie = False    # apply cv2 THRESH_TOZERO to max pixel values

    stimfile_dir = r'E:\research\ejhonglab\2021-11-30\3'
    thorimage_dir = os.path.join(stimfile_dir, 'flyfood')   # sometimes the folder is 'flyfood_00x' instead of 'flyfood'
    bounding_frame_yaml_cache = os.path.join(stimfile_dir, 'trial_bounding_frames.yaml')
    with open(bounding_frame_yaml_cache, 'r') as f:
        bounding_frames = yaml.safe_load(f)

    # load presentation order
    odor_list_unique_cache = os.path.join(stimfile_dir, 'odor_list_unique.p')
    odor_list_unique = pickle.load(open(odor_list_unique_cache, "rb"))

    movie = thor.read_movie(thorimage_dir)
    if filter_movie:
        movie = filtered(movie)
        fig_name = 'smooth_ROI.png'
    else:
        fig_name = 'ROI.png'

    # generate df/f movie
    dff_movie = np.zeros(shape=movie.shape)
    for presentation_index in range(len(bounding_frames)):
        start_frame, first_odor_frame, end_frame = bounding_frames[presentation_index]
        movie_baseline = movie[start_frame:(first_odor_frame - 1)].mean(axis=0)
        dff_movie[start_frame:(end_frame + 1)] = (movie[start_frame:(end_frame + 1)] - movie_baseline) / movie_baseline
    if write_dff_movie:
        util.write_tiff(os.path.join(thorimage_dir, 'dff_movie.tif'), dff_movie.astype(np.float32), strict_dtype=False)

    masks = util.ijroi_masks(os.path.join(thorimage_dir, 'RoiSet1.zip'), thorimage_dir)
    traces = pd.DataFrame(util.extract_traces_bool_masks(dff_movie, masks))
    roi_quality = traces.max() - traces.min()
    roi_nums, rois = util.rois2best_planes_only(masks, roi_quality)
    rois = rois.to_numpy()

    response_volumes = 3
    dff_ROI = {}
    for presentation_index in range(len(bounding_frames)):
        start_frame, first_odor_frame, end_frame = bounding_frames[presentation_index]
        movie_dff = dff_movie[first_odor_frame:first_odor_frame + response_volumes]
        movie_dff_max = np.amax(movie_dff, axis=0)

        # concatenate all pixels in ROI to a 1-d array
        values_in_roi = []
        [values_in_roi.append(movie_dff_max[rois[:, :, :, roi_index]]) for roi_index in range(rois.shape[3])]
        movie_dff_max_roi = np.concatenate(values_in_roi, axis=0)
        assert len(movie_dff_max_roi) == np.sum(rois)
        dff_ROI[odor_list_unique[presentation_index // 3] + str(presentation_index % 3)] = movie_dff_max_roi

        plot_correlation(dff_ROI, stimfile_dir, thresh_movie, sort_corr_mat, len(bounding_frames), odor_list_unique, fig_name)


if __name__ == '__main__':
    main()

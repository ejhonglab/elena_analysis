import os

from skimage.filters import gaussian
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from seaborn import clustermap

from hong2p import util, thor
from analysis import get_order

def main():
    write_dff_movie = False
    filter_movie = False

    stimfile_dir = r'E:\research\ejhonglab\2021-11-30\3'
    thorimage_dir = os.path.join(stimfile_dir, 'flyfood')
    bounding_frame_yaml_cache = os.path.join(stimfile_dir, 'trial_bounding_frames.yaml')
    odor_list_unique = ['control', 'ethyl acetate', 'benzaldehyde', 'cis-3-hexen-1-ol', 'butyric acid', '3-(methylthio)-1-propanol', 'flyfood', 'propionic acid', 'acetoin', 'isobutyric acid', 'acetic acid', 'ethanol', '-acetoin', '-propionic acid', '-ethyl acetate', '-butyric acid', '-cis-3-hexen-1-ol', '-isobutyric acid', '-acetic acid', '-3-(methylthio)-1-propanol', '-ethanol', '-benzaldehyde']
    with open(bounding_frame_yaml_cache, 'r') as f:
        bounding_frames = yaml.safe_load(f)


    movie = thor.read_movie(thorimage_dir)
    if filter_movie:
        for i in range(movie.shape[0]):
            for j in range(movie.shape[1]):
                movie[i, j, :, :] = cv2.GaussianBlur(movie[i, j, :, :], (5, 5), 0)

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

        values_in_roi = []
        [values_in_roi.append(movie_dff_max[rois[:, :, :, roi_index]]) for roi_index in range(rois.shape[3])]
        movie_dff_max_roi = np.concatenate(values_in_roi, axis=0)
        assert len(movie_dff_max_roi) == np.sum(rois)
        dff_ROI[odor_list_unique[presentation_index//3] + str(presentation_index % 3)] = movie_dff_max_roi

    dff_full_trial_df = pd.DataFrame(dff_ROI)

    odors_ind, odors = get_order()
    dff_full_trial_df = dff_full_trial_df.loc[:, odors_ind]
    dff_full_trial_legend = odors
    fig_name = 'ROI_correlation_sorted_filtered.png' if filter_movie else 'ROI_correlation_sorted.png'

    corr_mat = dff_full_trial_df.corr()

    fig, ax = plt.subplots()
    im = ax.imshow(corr_mat, vmax=np.amax(np.triu(corr_mat, 1)))

    ax.set_xticks(np.arange(len(bounding_frames)))
    ax.set_xticklabels(dff_full_trial_legend, rotation=90)
    ax.set_xticks(np.arange(1, len(bounding_frames), 3))

    ax.set_yticks(np.arange(len(bounding_frames)))
    ax.set_yticklabels(dff_full_trial_legend)
    ax.set_yticks(np.arange(1, len(bounding_frames), 3))

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(stimfile_dir, fig_name))

    scaler = StandardScaler()
    standardized = scaler.fit_transform(dff_full_trial_df.T)
    d = pd.DataFrame(standardized, columns=dff_full_trial_df.columns)

    pca = PCA()
    pca.fit(standardized)
    pca.explained_variance_ratio_


if __name__ == '__main__':
    main()

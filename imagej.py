#!/usr/bin/env python3

from os.path import join

import pandas as pd
import matplotlib.pyplot as plt

from hong2p import util, thor


def main():
    thorimage_dir = r'D:\research\ejhonglab\2021-11-09\1\flyfood'
    # roi_path = join(thorimage_dir, 'RoiSet.zip')

    movie = thor.read_movie(thorimage_dir)

    masks = util.ijroi_masks(thorimage_dir, thorimage_dir)

    # Would get one trace per ROI, without first merging ROIs by name
    # traces = util.extract_traces_bool_masks(movie, masks)

    # Groups single plane ROIs sharing name to form single volumetric ROI
    merged = util.merge_ijroi_masks(masks)
    traces = util.extract_traces_bool_masks(movie, merged)
    traces_each_plane = util.extract_traces_bool_masks(movie, masks)

    plt.imshow(movie[0, 0])

    plt.figure()

    plt.imshow(merged[0, :, :, 0])

    plt.show()

    import ipdb;
    ipdb.set_trace()


if __name__ == '__main__':
    main()

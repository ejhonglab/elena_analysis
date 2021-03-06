### Installation
```
pip install -r exact-requirements.txt
```

### AL Imaging
1. Convert raw movie to tiff: 
```
cd ~/flyfood
thor2tiff .
```
2. Draw ROI:
    - Open `raw.tif` in ImageJ, analyze -> tools -> ROI manager.
    - Draw one ROI around the antennal lobe for each plane.
    - Rename all ROIs to 'al'.
    - Save ROIs as `RoiSet.zip` in the `flyfood` folder.
4. Run `analysis.py`, this will generate the following in `stimfile_dir`: 
    - `merged_dff.svg`: weighted average df/f across merged ROIs; legend: 1 flyfood/1 control -> each component; 4 flyfood/4 control -> incomplete mixture from dropping 1 component; 5 flyfood/5 control -> complete mixture.
    - `x_correlation.png`: correlation calculated from maximum df/f during first 3 seconds after odor onset of each pixel in the ROI; `x` can be combinations of `smooth` `thresh` and `sorted` depending on your choice.
    - `single_trial_max_dffs.csv` and `multi_trial_max_dffs.csv` if `generate_csv == True`: maximum df/f traces during the first 3 seconds after odor onset; `single`: data for individual odor components; `multi`: data for complete and incomplete mixture.
    - `odor_list_unqiue.p`: odor presentation order without repeat.
    - `trial_bounding_frames.yaml`: start, odor onset, and end time frames of each odor trial
5. Run `heatmap.py`. This generates the maximum pixel df/f for each plane (x-axis) and repeat (y-axis) during the first 3 seconds after odor onset for each odor trial (title). Images are stored in the new folder `heatmap` in `stimfile_dir`.
6. Identify glomeruli: 
    - Install [this](https://github.com/ejhonglab/imagej_macros).
    - Draw one ROI around each glomerulus.
    - Rename so that ROIs with the same glomerulus have the same name.
    - Save ROIs as `RoiSet1.zip` in the `flyfood` folder.
8. Run `ROI.py`, this will generate the following: 
    - `dff_movie.tif` in `flyfood` if `write_dff_movie == True`: df/f movie.
    - `x_ROI.png`: correlation calculated from maximum df/f during first 3 seconds after odor onset of each pixel in the glomeruli ROIs; `x` can be combinations of `smooth` `thresh` and `sorted` depending on your choice.
9. Run `plot_stats.ipynb` in jupyter notebook. This will plot the maximum df/f traces for each odor trial for all experiments performed on the same day. 
#### Example data
HongLab @ Caltech Dropbox/Rotation/Elena Fall 2021/2021-11-30/3

### Code not related to AL imaging
1. `orn_distance.py`: code for data analysis on Hallem dataset with fruit samples, `HC_data_raw.csv`.
2. `control_mixture.ipynb`: select control mixture based on pmi, correlation, and water solubility; data folder: HongLab @ Caltech/Rotation/Elena Fall 2021/data

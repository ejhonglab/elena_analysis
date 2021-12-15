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
2. Draw ROI: 1) open raw.tif in ImageJ, analyze -> tools -> ROI manager; 2) draw one ROI around the antennal lobe for each plane; 3) rename all ROIs to 'al'; 4) save ROIs as `RoiSet.zip` in the `flyfood` folder.
3. Run `analysis.py`, this will generate the following in `stimfile_dir`: 
    - `merged_dff.svg`: weighted average df/f across merged ROIs; legend: 1 flyfood/1 control -> each component; 4 flyfood/4 control -> incomplete mixture from dropping 1 component; 5 flyfood/5 control -> complete mixture.
    - `x_correlation.png`: correlation calculated from maximum df/f during first 3 seconds after odor onset of each pixel in the ROI; `x` can be combinations of `smooth` `thresh` and `sorted` depending on your choice.
    - `single_trial_max_dffs.csv` and `multi_trial_max_dffs.csv` if `generate_csv == True`: maximum df/f traces during the first 3 seconds after odor onset; `single`: data for individual odor components; `multi`: data for complete and incomplete mixture.
    - `odor_list_unqiue.p`: odor presentation order without repeat.
    - `trial_bounding_frames.yaml`: start, odor onset, and end time frames of each odor trial
4. Run `heatmap.py`. This generates the maximum pixel df/f for each plane (x-axis) and repeat (y-axis) during the first 3 seconds after odor onset for each odor trial (title). Images are stored in the new folder `heatmap` in `stimfile_dir`.
5. Identify glomeruli: 1) use https://github.com/ejhonglab/imagej_macros; 2) draw one ROI around each glomerulus; 3) rename so that ROIs with the same glomerulus have the same name; 4) save ROIs as `RoiSet1.zip` in the `flyfood` folder.
6. Run `ROI.py`, this will generate the following: 
    - `dff_movie.tif` in `flyfood` if `write_dff_movie == True`: df/f movie.
    - `x_ROI.png`: correlation calculated from maximum df/f during first 3 seconds after odor onset of each pixel in the glomeruli ROIs; `x` can be combinations of `smooth` `thresh` and `sorted` depending on your choice.
7. Run `plot_stats.ipynb` in jupyter notebook. This will plot the maximum df/f traces for each odor trial for all experiments performed on the same day. 

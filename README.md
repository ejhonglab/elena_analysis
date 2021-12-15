### Installation
```
pip install -r exact-requirements.txt
```

### Running
1. Convert raw movie to tiff: 
```
cd ~/flyfood
thor2tiff .
```
2. Draw ROI: 1) open raw.tif in ImageJ, analyze -> tools -> ROI manager; 2) draw one ROI around the antennal lobe for each plane; 3) rename all ROIs to 'al'; 4) save ROIs as `RoiSet` in the `flyfood` folder.
3. Run `analysis.py`, this will generate the following in `stimfile_dir`: 
  - `merged_dff.svg`: df/f trace across merged ROIs; legend: 1 flyfood/1 control -> each component; 4 flyfood/4 control -> incomplete mixture from dropping 1 component; 5 flyfood/5 control -> complete mixture.
  - `x_correlation.png`: pixel correlation; `x` can be combinations of `smooth` `thresh` and `sort` depending on your choice.
  - `single_trial_max_dffs.csv` and `multi_trial_max_dffs.csv` if `generate_csv == True`: maximum of df/f traces during the first 3 seconds after odor onset; `single_trial_max_dffs.csv` -> data for individual odor components; `multi_trial_max_dffs.csv` -> data for complete and incomplete mixture.
  - `odor_list_unqiue.p`: odor presentation order without repeat
  - `trial_bounding_frames.yaml`

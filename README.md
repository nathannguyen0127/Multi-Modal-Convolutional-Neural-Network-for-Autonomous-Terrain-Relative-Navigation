
# Multi-Modal Convolutional Neural Network for Autonomous Terrain Relative Navigation
## by Nathan Nguyen 
This repository implements a **multi-level multi-modal convolutional neural network (MMCNN)** cascade for autonomous **Terrain Relative Navigation (TRN)**. The pipeline supports:

- Full training of hierarchical MMCNN models (Level 1, 2, 3)
- Model saving/loading for reproducibility
- Inference and error evaluation using joint probability
- Temperature scaling calibration
- Visualization of predicted vs. ground-truth positions

## Dataset
The dataset was generated synthetically via Blender and a digital elevation model of a small section of the area near the Apollo 16 landing site. Accesible via this link: https://wms.lroc.asu.edu/lroc/view_rdr/NAC_DTM_APOLLO16

Images from Blender will go to:
```
repo/data/Outputs
```
## MMCNN Overview
The **multi-level multi-modal convolutional neural network (MMCNN)** cascade for autonomous **Terrain Relative Navigation (TRN)** essentially breaks down a conventional CNN regression to a CNN classification of quadrants into a probabilistic regression.

### Model Details
- **Level 1**: 128×128 input, predicts coarse 2×2 quadrant
- **Level 2**: 64×64 subregion, refines to 4 quadrants within L1
- **Level 3**: 32×32 subregion, final 8×8 resolution
- Uses **intensity and depth fusion** via parallel branches

## Directory Structure

```
repo/
├── data/                           
│   ├── Outputs/                    # Images and EXRs from Blender CHECK ONEDRIVE LINK BELOW
│   └── labeled_dataset.pkl.py      # Dataset generated by generate_dataset
├── results/                        # Model and metadata outputs   CHECK ONEDRIVE LINK BELOW
├── src/
│   ├── create.py                   # Randomly set camera pose array
│   ├── lunar_terrain.blend         # Blender file of Lunar Terrain
│   ├── script.py                   # Blender Python script images in Output/
│   ├── run_script.py               # Runs script.py
│   ├── generate_dataset_funcs.py   # Dataset generation and labeling
│   ├── calibration.py              # Post-hoc temperature scaling
│   ├── evaluation.py               # Joint probability evaluation and plotting
│   ├── inference.py                # Hierarchical probability inference
│   ├── model.py                    # MMCNN architecture and wrapper
│   ├── training.py                 # Full cascade training pipeline
│   └── dataset.py                  # Data preprocessing and splitting for training
├── notebooks/
│   ├── generate_dataset.py         # Pipeline for dataset generation and labeling
│   └── main_pipeline.py            # Entry-point for training, evaluation, and visualization
```

## Quick Start

### 1. Install Dependencies

Make sure you have Python 3.8+ and install the required packages using requirements.txt:

### 2. Prepare Your Dataset

Place your preprocessed dataset in:

```
repo/data/labeled_dataset.pkl
```

The DataFrame should contain the following columns:
- `image` : 2D numpy array (128×128 intensity image)
- `depth` : 2D numpy array (128×128 depth map)
- `crop_size`, `level1_quadrant`, `level2_quadrant`, `level3_quadrant`
- `center_x`, `center_y`: ground truth pixel positions

### 3. Run the Full Pipeline

From the `repo/notebooks/` folder:

```bash
python main_pipeline.py
```

This will:
- Train the full MMCNN cascade
- Save all models to `results/`
- Apply temperature scaling
- Evaluate uncalibrated and calibrated models
- Save a heatmap image (`heatmap_prediction_sample_0.png`)

### 4. Skip Training and Just Evaluate

If you’ve already trained and saved the models:

```python
# In main_pipeline.py
run_pipeline(df, training=False)
```
## Output

- `results/L1.h5`, `L2_*.h5`, `L3_*.h5`: Trained models
- `results/metadata_*.pkl`: Test data split metadata
- `results/heatmap_prediction_sample_0.png`: Visualization of one prediction

## Results
The model trains well and even slightly better in some cases when temperature-scaled. Th
Example evaluation:
Evaluated 100 samples
Mean Euclidean Error: 54.06 m
Std Deviation: 35.86 m
Max Error: 166.93 m

Under these metrics we can see promising results where the lander has amean error of approximately 54 meters durings its landing. Some samples can be shown in the graph
## Notes
- You must implement `prepare_dataset()` in `src/dataset.py` to return:
  ```python
  X_train, X_val, y_train, y_val, X_test, y_test, df_test
  ```
- Temperature scaling is applied only to Level 1 for post-hoc calibration.

data for /data/
https://rpiexchange-my.sharepoint.com/:f:/g/personal/nguyen2_rpi_edu/EnDF-KXA25tNpJZ9-9Y871wBR00DAxYYZn4rCE-fULGIsA?e=cPHwLY

models for /results/
https://rpiexchange-my.sharepoint.com/:f:/g/personal/nguyen2_rpi_edu/Eg7QoZ6jHndHiozcHfu2dP8B2TTyBW8rxy-EmeZTNJloIg?e=5oqQiZ



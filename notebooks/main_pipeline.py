import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from training import train_mmcnn_cascade_full
from model import TemperatureScaledModel
from calibration import tune_temperature, apply_temperature_scaling
from evaluation import evaluate_model_overall, compute_expected_position
from inference import infer_joint_probability

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

def save_models(models):
    """
    Saves trained models from dictionary into RESULTS_DIR.
    """
    for level, content in models.items():
        if level == 'L1':
            models[level].save(os.path.join(RESULTS_DIR, f"L1.h5"))
        elif level == 'L2':
            for l1, model in content.items():
                model.save(os.path.join(RESULTS_DIR, f"L2_{l1}.h5"))
        elif level == 'L3':
            for (l1, l2), model in content.items():
                model.save(os.path.join(RESULTS_DIR, f"L3_{l1}_{l2}.h5"))

def save_calibrated_models(cal_models):
    """
    Saves all calibrated models to RESULTS_DIR with `.keras` extension.

    Args:
        cal_models (dict): Dictionary of calibrated TemperatureScaledModel objects.
    """
    if 'L1' in cal_models:
        cal_models['L1'].save(os.path.join(RESULTS_DIR, "L1_calibrated.keras"))

    for l1 in cal_models.get('L2', {}):
        cal_models['L2'][l1].save(os.path.join(RESULTS_DIR, f"L2_{l1}_calibrated.keras"))

    for (l1, l2), model in cal_models.get('L3', {}).items():
        model.save(os.path.join(RESULTS_DIR, f"L3_{l1}_{l2}_calibrated.keras"))

def load_models():
    """
    Loads MMCNN model files from RESULTS_DIR.

    Returns:
        dict: Dictionary with keys 'L1', 'L2', 'L3'.
    """
    models = {'L2': {}, 'L3': {}}
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".h5"):
            parts = file.replace(".h5", "").split("_")
            model_path = os.path.join(RESULTS_DIR, file)
            if parts[0] == "L1":
                models['L1'] = tf.keras.models.load_model(model_path)
            elif parts[0] == "L2":
                models['L2'][int(parts[1])] = tf.keras.models.load_model(model_path)
            elif parts[0] == "L3":
                l1, l2 = int(parts[1]), int(parts[2])
                models['L3'][(l1, l2)] = tf.keras.models.load_model(model_path)
    return models

def load_calibrated_models():
    """
    Loads calibrated TemperatureScaledModel instances from RESULTS_DIR.

    Returns:
        dict: Dictionary of calibrated models {'L1': ..., 'L2': {...}, 'L3': {...}}
    """
    models = {'L2': {}, 'L3': {}}

    # Load L1
    l1_path = os.path.join(RESULTS_DIR, "L1_calibrated.keras")
    if os.path.exists(l1_path):
        models['L1'] = tf.keras.models.load_model(l1_path, custom_objects={'TemperatureScaledModel': TemperatureScaledModel})
        print("Loaded calibrated L1 model.")
    else:
        raise FileNotFoundError("L1_calibrated.keras not found.")

    # Load L2
    for l1 in range(1, 5):
        l2_path = os.path.join(RESULTS_DIR, f"L2_{l1}_calibrated.keras")
        if os.path.exists(l2_path):
            models['L2'][l1] = tf.keras.models.load_model(l2_path, custom_objects={'TemperatureScaledModel': TemperatureScaledModel})

    # Load L3
    for l1 in range(1, 5):
        for l2 in range(1, 5):
            l3_path = os.path.join(RESULTS_DIR, f"L3_{l1}_{l2}_calibrated.keras")
            if os.path.exists(l3_path):
                models['L3'][(l1, l2)] = tf.keras.models.load_model(l3_path, custom_objects={'TemperatureScaledModel': TemperatureScaledModel})

    return models

def save_metadata(metadata):
    """
    Saves metadata dictionary as pickle files in RESULTS_DIR.
    """
    for level, content in metadata.items():
        if level == 'L1':
            content.to_pickle(os.path.join(RESULTS_DIR, "metadata_L1.pkl"))
        elif level == 'L2':
            for l1, df in content.items():
                df.to_pickle(os.path.join(RESULTS_DIR, f"metadata_L2_{l1}.pkl"))
        elif level == 'L3':
            for (l1, l2), df in content.items():
                df.to_pickle(os.path.join(RESULTS_DIR, f"metadata_L3_{l1}_{l2}.pkl"))

def load_metadata():
    """
    Loads metadata pickle files from RESULTS_DIR.

    Returns:
        dict: Dictionary with structure matching that from training.
    """
    metadata = {'L1': None, 'L2': {}, 'L3': {}}
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".pkl") and file.startswith("metadata"):
            parts = file.replace("metadata_", "").replace(".pkl", "").split("_")
            file_path = os.path.join(RESULTS_DIR, file)
            if parts[0] == "L1":
                metadata['L1'] = pd.read_pickle(file_path)
            elif parts[0] == "L2":
                metadata['L2'][int(parts[1])] = pd.read_pickle(file_path)
            elif parts[0] == "L3":
                l1, l2 = int(parts[1]), int(parts[2])
                metadata['L3'][(l1, l2)] = pd.read_pickle(file_path)
    return metadata

def save_predictions(predictions):
    with open(os.path.join(RESULTS_DIR, "predictions.pkl"), "wb") as f:
        pickle.dump(predictions, f)

def load_predictions():
    with open(os.path.join(RESULTS_DIR, "predictions.pkl"), "rb") as f:
        return pickle.load(f)


def save_joint_probability_map(joint_probs, gt_x, gt_y, pred_x, pred_y, save_path):
    """
    Saves the 8x8 heatmap of joint probabilities to disk with GT and prediction overlaid.

    Args:
        joint_probs (np.ndarray): (4, 4, 4) joint probability cube.
        gt_x, gt_y (float): Ground truth coordinates.
        pred_x, pred_y (float): Predicted coordinates.
        save_path (str): Output file path for saved heatmap image.
    """
    grid = np.zeros((8, 8))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                l1_row, l1_col = i // 2, i % 2
                l2_row, l2_col = j // 2, j % 2
                l3_row, l3_col = k // 2, k % 2
                row = (l1_row * 2 + l2_row) * 2 + l3_row
                col = (l1_col * 2 + l2_col) * 2 + l3_col
                grid[row, col] += joint_probs[i, j, k]

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='hot', interpolation='nearest', extent=[96, 672, 672, 96])
    plt.colorbar(label='Joint Probability')
    plt.scatter(gt_x, gt_y, color='green', label='Ground Truth', marker='x', s=100)
    plt.scatter(pred_x, pred_y, color='cyan', label='Predicted', marker='o', s=100)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(False)
    plt.title("Prediction Heatmap")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def run_pipeline(df, training = True, calibrating = True):
    """
    Full training, saving, reloading, and evaluation pipeline.
    """
    if training:
        print("Training MMCNN cascade...")
        models, predictions, metadata = train_mmcnn_cascade_full(df)

        print("Saving models and metadata...")
        save_models(models)
        save_metadata(metadata)
        save_predictions(predictions)
    else:
        print("Reloading saved models...")
        models = load_models()
        metadata = load_metadata()
        predictions = load_predictions()

    print("Evaluating baseline performance...")
    errors_raw, _, _ = evaluate_model_overall(metadata['L1'], models, method="mean")

    if calibrating:
        print("Applying temperature scaling to all model levels...")
        cal_models = apply_temperature_scaling(models, predictions)

        print("Saving calibrated models...")
        save_calibrated_models(cal_models)
    else:
        print("Reloading saved calibrated models...")
        cal_models = load_calibrated_models()

    print("Evaluating calibrated model...")
    errors_cal, _, _ = evaluate_model_overall(metadata['L1'], cal_models, method="mean")

    #Visualize and save heatmap for a single sample
    print("Saving prediction heatmap for one sample...")
    sample = metadata['L1'].iloc[141]
    img = sample['image']
    dpt = sample['depth']
    gt_x, gt_y = sample['center_x'], sample['center_y']

    joint_probs = infer_joint_probability(
        img / 65535.0,
        (dpt - np.min(dpt)) / max(np.max(dpt) - np.min(dpt), 1e-5),
        cal_models
    )
    pred_x, pred_y = compute_expected_position(joint_probs)

    heatmap_path = os.path.join(RESULTS_DIR, "heatmap_prediction_sample_0.png")
    save_joint_probability_map(joint_probs, gt_x, gt_y, pred_x, pred_y, heatmap_path)

    print(f"Heatmap saved to {heatmap_path}")


if __name__ == "__main__":
    df_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled_dataset.pkl'))
    df = pd.read_pickle(df_path)
    run_pipeline(df, training=False, calibrating=True)

    
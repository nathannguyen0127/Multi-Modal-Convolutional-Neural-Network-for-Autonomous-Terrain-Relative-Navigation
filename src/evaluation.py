import numpy as np
import matplotlib.pyplot as plt
from inference import get_joint_probabilities_from_center, get_joint_probabilities_mean

def compute_expected_position(joint_probs, base_origin=(96, 96), cell_size=72):
    """
    Computes the expected (x, y) position from a joint probability cube over 64 spatial cells.

    Args:
        joint_probs (np.ndarray): (4, 4, 4) joint probability cube.
        base_origin (tuple): (x, y) origin of the 576x576 central region.
        cell_size (int): Side length of each Level 3 cell.

    Returns:
        (float, float): Expected (x, y) coordinates.
    """
    expected_x, expected_y = 0.0, 0.0
    total_prob = np.sum(joint_probs)

    for i in range(4):
        for j in range(4):
            for k in range(4):
                prob = joint_probs[i, j, k]
                if prob == 0:
                    continue

                l1_row, l1_col = i // 2, i % 2
                l2_row, l2_col = j // 2, j % 2
                l3_row, l3_col = k // 2, k % 2
                row = (l1_row * 2 + l2_row) * 2 + l3_row
                col = (l1_col * 2 + l2_col) * 2 + l3_col

                center_x = base_origin[0] + col * cell_size + cell_size / 2
                center_y = base_origin[1] + row * cell_size + cell_size / 2
                expected_x += prob * center_x
                expected_y += prob * center_y

    if total_prob > 0:
        expected_x /= total_prob
        expected_y /= total_prob
    else:
        expected_x, expected_y = -1, -1

    return expected_x, expected_y


def plot_joint_probability_map(joint_probs, gt_x, gt_y, pred_x, pred_y):
    """
    Visualizes the 8x8 heatmap of marginal joint probability and overlays predicted vs. true position.

    Args:
        joint_probs (np.ndarray): (4, 4, 4) joint probabilities.
        gt_x, gt_y (float): Ground-truth coordinates.
        pred_x, pred_y (float): Predicted coordinates.
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
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from evaluation import compute_expected_position
from inference import get_joint_probabilities_from_center, get_joint_probabilities_mean


def evaluate_model_overall(df, models, method="centered", max_samples=100):
    """
    Evaluates MMCNN models over a batch of image-depth pairs.

    Args:
        df (pd.DataFrame): Test dataset with image, depth, and ground truth center.
        models (dict): Dictionary of trained models.
        method (str): One of 'centered' or 'mean'. Controls joint probability strategy.
        max_samples (int): Maximum number of samples to evaluate.

    Returns:
        Tuple (errors, predictions, ground_truths) where:
            errors (list): Euclidean error per sample
            predictions (list): (x, y) predicted positions
            ground_truths (list): (x, y) true positions
    """
    errors = []
    preds = []
    gts = []

    for i, row in df.iterrows():
        image = row["image"]
        depth = row["depth"]
        true_x = row["center_x"]
        true_y = row["center_y"]

        if method == "centered":
            joint_probs = get_joint_probabilities_from_center(image, depth, models)
        elif method == "mean":
            joint_probs = get_joint_probabilities_mean(image, depth, models)
        else:
            raise ValueError("Invalid method: choose 'centered' or 'mean'")

        pred_x, pred_y = compute_expected_position(joint_probs)
        error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)

        errors.append(error)
        preds.append((pred_x, pred_y))
        gts.append((true_x, true_y))

        if len(errors) >= max_samples:
            break

    errors = np.array(errors)
    print(f"Evaluated {len(errors)} samples")
    print(f"Mean Euclidean Error: {np.mean(errors):.2f} m")
    print(f"Std Deviation: {np.std(errors):.2f} m")
    print(f"Max Error: {np.max(errors):.2f} m")

    return errors, preds, gts

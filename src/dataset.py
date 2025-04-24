import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset(df, crop_size=64, target_col='level2_quadrant'):
    """
    Prepares a dataset of image-depth pairs and corresponding labels for MMCNN training.
    Images and depth maps are normalized and stacked into a 2-channel format.

    Args:
        df (pd.DataFrame): Full labeled dataset containing images, depths, and quadrant labels.
        crop_size (int): Resolution of image crops (128, 64, or 32).
        target_col (str): Column name for label (e.g., 'level2_quadrant').

    Returns:
        Tuple of train/val/test splits for X, y, and test metadata (df_test).
    """
    df_sub = df[df["crop_size"] == crop_size].copy()

    X_intensity = np.stack([
        img.astype(np.float32) / 65535.0 for img in df_sub["image"].values
    ])

    X_depth = []
    for d, i in zip(df_sub["depth"], df_sub["image"]):
        if d is not None:
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max - d_min < 1e-5:
                X_depth.append(np.zeros_like(d))
            else:
                X_depth.append((d - d_min) / (d_max - d_min))
        else:
            X_depth.append(np.zeros_like(i))

    X_depth = np.stack(X_depth)
    X = np.stack([np.stack([i, d], axis=-1) for i, d in zip(X_intensity, X_depth)])
    y = (df_sub[target_col].values - 1).astype(np.int32)

    # Stratified split into train, val, test
    X_train_full, X_val, y_train_full, y_val, df_train_full, df_val = train_test_split(
        X, y, df_sub, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_train_full, y_train_full, df_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    return X_train, X_val, y_train, y_val, X_test, y_test, df_test
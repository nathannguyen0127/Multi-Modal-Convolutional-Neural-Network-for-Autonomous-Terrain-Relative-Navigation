import numpy as np

def infer_joint_probability(image_128_norm, depth_128_norm, models):
    """
    Computes the joint (L1, L2, L3) probability cube for a given normalized image-depth input.

    Args:
        image_128_norm (np.ndarray): Normalized intensity image (128x128).
        depth_128_norm (np.ndarray): Normalized depth map (128x128).
        models (dict): Dictionary containing trained MMCNNs for each level.

    Returns:
        np.ndarray: Joint probability cube of shape (4, 4, 4).
    """
    def stack_input(img, dpt):
        return np.stack([img, dpt], axis=-1)[np.newaxis, ...]

    joint_probs = np.zeros((4, 4, 4))
    input_128 = stack_input(image_128_norm, depth_128_norm)
    p1 = models['L1'].predict(input_128, verbose=0)[0]

    for i in range(4):
        row64 = (i // 2) * 64
        col64 = (i % 2) * 64
        img_64 = image_128_norm[row64:row64+64, col64:col64+64]
        dep_64 = depth_128_norm[row64:row64+64, col64:col64+64]
        input_64 = stack_input(img_64, dep_64)

        if (i + 1) not in models['L2']:
            continue
        p2 = models['L2'][i + 1].predict(input_64, verbose=0)[0]

        for j in range(4):
            row32 = row64 + (j // 2) * 32
            col32 = col64 + (j % 2) * 32
            img_32 = image_128_norm[row32:row32+32, col32:col32+32]
            dep_32 = depth_128_norm[row32:row32+32, col32:col32+32]
            input_32 = stack_input(img_32, dep_32)

            if (i + 1, j + 1) not in models['L3']:
                continue
            p3 = models['L3'][(i + 1, j + 1)].predict(input_32, verbose=0)[0]

            for k in range(4):
                joint_probs[i, j, k] = p1[i] * p2[j] * p3[k]

    return joint_probs

def get_joint_probabilities_from_center(image_128, depth_128, models):
    def preprocess(img, dpt):
        i = img.astype(np.float32) / 65535.0
        d_min, d_max = np.min(dpt), np.max(dpt)
        d = (dpt - d_min) / (d_max - d_min) if d_max - d_min > 1e-5 else np.zeros_like(dpt)
        return np.stack([i, d], axis=-1)[np.newaxis, ...]

    input_128 = preprocess(image_128, depth_128)
    p1 = models['L1'].predict(input_128)[0]
    l1_idx = np.argmax(p1)

    row64 = (l1_idx // 2) * 64
    col64 = (l1_idx % 2) * 64
    img_64 = image_128[row64:row64+64, col64:col64+64]
    dep_64 = depth_128[row64:row64+64, col64:col64+64]
    input_64 = preprocess(img_64, dep_64)

    if l1_idx + 1 not in models['L2']:
        return np.zeros((4, 4, 4))

    p2 = models['L2'][l1_idx + 1].predict(input_64)[0]
    l2_idx = np.argmax(p2)

    row32 = row64 + (l2_idx // 2) * 32
    col32 = col64 + (l2_idx % 2) * 32
    img_32 = image_128[row32:row32+32, col32:col32+32]
    dep_32 = depth_128[row32:row32+32, col32:col32+32]
    input_32 = preprocess(img_32, dep_32)

    key = (l1_idx + 1, l2_idx + 1)
    if key not in models['L3']:
        return np.zeros((4, 4, 4))

    p3 = models['L3'][key].predict(input_32)[0]
    l3_idx = np.argmax(p3)

    # --- Assign only one active triplet to joint probability ---
    joint_probs = np.zeros((4, 4, 4))
    joint_probs[l1_idx, l2_idx, l3_idx] = p1[l1_idx] * p2[l2_idx] * p3[l3_idx]

    return joint_probs

def get_joint_probabilities_mean(image_128, depth_128, models):
    def preprocess(img, dpt):
        i = img.astype(np.float32) / 65535.0
        d = dpt.astype(np.float32)
        d = (d - np.min(d)) / (np.max(d) - np.min(d)) if np.max(d) - np.min(d) > 1e-5 else np.zeros_like(d)
        return np.stack([i, d], axis=-1)[np.newaxis, ...]

    input_128 = preprocess(image_128, depth_128)
    p1 = models['L1'].predict(input_128)[0]

    joint_probs = np.zeros((4, 4, 4))

    for i in range(4):
        r64, c64 = (i // 2) * 64, (i % 2) * 64
        img_64 = image_128[r64:r64+64, c64:c64+64]
        dpt_64 = depth_128[r64:r64+64, c64:c64+64]
        input_64 = preprocess(img_64, dpt_64)

        if i+1 not in models['L2']:
            continue
        p2 = models['L2'][i+1].predict(input_64)[0]

        for j in range(4):
            r32, c32 = r64 + (j // 2) * 32, c64 + (j % 2) * 32
            img_32 = image_128[r32:r32+32, c32:c32+32]
            dpt_32 = depth_128[r32:r32+32, c32:c32+32]
            input_32 = preprocess(img_32, dpt_32)

            if (i+1, j+1) not in models['L3']:
                continue
            p3 = models['L3'][(i+1, j+1)].predict(input_32)[0]

            for k in range(4):
                joint_probs[i, j, k] = p1[i] * p2[j] * p3[k]

    return joint_probs

import numpy as np
import OpenEXR, Imath
import re

def load_exr_depth(filepath):
    """
    Load a single-channel depth image from an OpenEXR file.

    Parameters:
        filepath (str): Path to the EXR file.

    Returns:
        np.ndarray: 2D array representing depth.
    """
    exr = OpenEXR.InputFile(filepath)
    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr.channel('V', FLOAT)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
    return depth

def split_into_quadrants(img, size):
    """
    Split a square image into non-overlapping square crops of the given size.

    Parameters:
        img (np.ndarray): Input image (H x W).
        size (int): Size of each square crop.

    Returns:
        list: Tuples of form (row, col, cropped_image).
    """
    h, w = img.shape
    assert h % size == 0 and w % size == 0
    return [(i, j, img[i:i+size, j:j+size]) for i in range(0, h, size) for j in range(0, w, size)]

def extract_center_from_filename(filename):
    """
    Extract the center coordinates from a file name.

    Parameters:
        filename (str): File name containing coordinates.

    Returns:
        tuple: (x, y) coordinates as floats.
    """
    match = re.search(r'_([\d.]+)_([\d.]+)_', filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def get_multilevel_quadrants(x, y):
    """
    Get hierarchical quadrant labels (L1, L2, L3) for a given (x, y) coordinate in 768x768 image space.

    Returns:
        tuple: (L1, L2, L3) quadrant labels.
    """
    if not (96 <= x < 672 and 96 <= y < 672):
        return 0, 0, 0

    x_c = x - 96
    y_c = y - 96

    l1_col = 0 if x_c < 288 else 1
    l1_row = 0 if y_c < 288 else 1
    l1_quad = 1 + l1_col + 2 * l1_row

    x_l2 = x_c % 288
    y_l2 = y_c % 288
    l2_col = 0 if x_l2 < 144 else 1
    l2_row = 0 if y_l2 < 144 else 1
    l2_quad = 1 + l2_col + 2 * l2_row

    x_l3 = x_l2 % 144
    y_l3 = y_l2 % 144
    l3_col = 0 if x_l3 < 72 else 1
    l3_row = 0 if y_l3 < 72 else 1
    l3_quad = 1 + l3_col + 2 * l3_row

    return l1_quad, l2_quad, l3_quad

def extract_camera_center_key(filename):
    """
    Extract camera index and center coordinates rounded to three decimals from a file name.

    Parameters:
        filename (str): File name containing camera and coordinates.

    Returns:
        tuple: (camera_index, x, y) or None if parsing fails.
    """
    match = re.search(r'Camera (\d+)_([\d.]+)_([\d.]+)', filename)
    if match:
        cam = int(match.group(1))
        x = round(float(match.group(2)), 3)
        y = round(float(match.group(3)), 3)
        return (cam, x, y)
    return None

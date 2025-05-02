"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog


def detect_edges(image: Image.Image, method: str = "canny") -> Image.Image:
    """
    Detects edges in an image using the specified method.

    Args:
        image (Image.Image): The input image.
        method (str): The edge detection method ('canny' or 'sobel').

    Returns:
        Image.Image: The image with detected edges.
    """
    image_np = np.array(image.convert("L"))

    if method == "canny":
        edges = cv2.Canny(image_np, 100, 200)
    elif method == "sobel":
        sobelx = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(edges)
    else:
        raise ValueError("Method should be 'canny' or 'sobel'.")

    return Image.fromarray(edges)


def calculate_color_histogram(image: Image.Image) -> np.ndarray:
    """
    Calculates the color histogram of an image.

    Args:
        image (Image.Image): The input image.

    Returns:
        np.ndarray: The color histogram.
    """
    image_np = np.array(image)
    histogram = []

    for channel in range(3):
        hist = cv2.calcHist([image_np], [channel], None, [256], [0, 256])
        histogram.append(hist)

    histogram = np.concatenate(histogram, axis=1)
    return histogram


def detect_keypoints(image: Image.Image, method: str = "sift") -> Image.Image:
    """
    Detects keypoints and extracts descriptors using the specified method.

    Args:
        image (Image.Image): The input image.
        method (str): The keypoint detection method ('sift', 'surf', or 'orb').

    Returns:
        Image.Image: The image with keypoints drawn.
    """
    image_np = np.array(image.convert("L"))

    if method == "sift":
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image_np, None)
    elif method == "surf":
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(image_np, None)
    elif method == "orb":
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image_np, None)
    else:
        raise ValueError("Method should be 'sift', 'surf', or 'orb'.")

    keypoint_image = cv2.drawKeypoints(image_np, keypoints, None)
    return Image.fromarray(keypoint_image)


def extract_hog_features(image: Image.Image) -> np.ndarray:
    """
    Extracts Histogram of Oriented Gradients (HOG) features from an image.

    Args:
        image (Image.Image): The input image.

    Returns:
        np.ndarray: The HOG features.
    """
    image_np = np.array(image.convert("L"))
    hog_features, hog_image = hog(
        image_np, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True
    )

    return hog_features, Image.fromarray(hog_image)

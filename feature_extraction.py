"""
Feature Extraction Module for AI vs Real Image Classification
Extracts rich feature vectors from images for ML models
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel
from scipy.fftpack import fft2, fftshift
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')


def extract_color_histogram(image, bins=32):
    """Extract color histogram features from BGR image."""
    features = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
    return np.array(features)


def extract_hsv_features(image):
    """Extract HSV color space statistics."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    for channel in range(3):
        ch = hsv[:, :, channel].flatten()
        features.extend([
            np.mean(ch), np.std(ch), skew(ch), kurtosis(ch),
            np.percentile(ch, 25), np.percentile(ch, 75)
        ])
    return np.array(features)


def extract_texture_features(gray_image):
    """Extract GLCM texture features."""
    gray_uint8 = (gray_image * 255).astype(np.uint8) if gray_image.max() <= 1 else gray_image.astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        feat = graycoprops(glcm, prop).flatten()
        features.extend(feat)
    return np.array(features)


def extract_lbp_features(gray_image, radius=3, n_points=24):
    """Extract Local Binary Pattern histogram."""
    gray_uint8 = gray_image.astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    return hist


def extract_edge_features(gray_image):
    """Extract edge-based features using Sobel and Canny."""
    gray_float = gray_image.astype(np.float64) / 255.0
    sobel_edges = sobel(gray_float)
    
    gray_uint8 = gray_image.astype(np.uint8)
    canny_edges = cv2.Canny(gray_uint8, 50, 150)
    
    features = [
        np.mean(sobel_edges), np.std(sobel_edges),
        np.mean(canny_edges), np.sum(canny_edges > 0) / canny_edges.size,
        np.percentile(sobel_edges, 90), np.percentile(sobel_edges, 99)
    ]
    return np.array(features)


def extract_frequency_features(gray_image):
    """Extract frequency domain features via FFT."""
    f_transform = fft2(gray_image.astype(np.float64))
    f_shifted = fftshift(f_transform)
    magnitude = np.abs(f_shifted)
    magnitude_log = np.log1p(magnitude)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radii = [10, 30, 60, 100]
    Y, X = np.ogrid[:h, :w]
    features = []
    prev_mask = np.zeros((h, w), dtype=bool)
    
    for r in radii:
        mask = ((X - cx)**2 + (Y - cy)**2 <= r**2)
        ring = mask & ~prev_mask
        features.append(magnitude_log[ring].mean() if ring.sum() > 0 else 0)
        prev_mask = mask
    
    # High frequency energy ratio
    high_freq_mask = ~prev_mask
    total_energy = magnitude_log.sum()
    hf_energy = magnitude_log[high_freq_mask].sum() / (total_energy + 1e-10)
    features.append(hf_energy)
    return np.array(features)


def extract_noise_features(gray_image):
    """Extract noise-related features — AI images often have distinct noise patterns."""
    gray_float = gray_image.astype(np.float64)
    blurred = cv2.GaussianBlur(gray_float, (5, 5), 0)
    noise = gray_float - blurred
    
    features = [
        np.mean(noise), np.std(noise),
        kurtosis(noise.flatten()), skew(noise.flatten()),
        np.percentile(np.abs(noise), 95)
    ]
    return np.array(features)


def extract_saturation_uniformity(image):
    """AI images tend to have more uniform saturation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    saturation = hsv[:, :, 1]
    
    # Compute local variance of saturation
    kernel = np.ones((8, 8), np.float32) / 64
    local_mean = cv2.filter2D(saturation.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D((saturation**2).astype(np.float32), -1, kernel)
    local_var = local_sq_mean - local_mean**2
    
    features = [
        np.mean(local_var), np.std(local_var),
        np.mean(saturation), np.std(saturation),
        np.percentile(local_var, 90)
    ]
    return np.array(features)


def extract_all_features(image_path, target_size=(128, 128)):
    """
    Extract complete feature vector from an image file.
    Returns a 1D numpy array of features.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    image = cv2.resize(image, target_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = np.concatenate([
        extract_color_histogram(image, bins=32),    # 96 features
        extract_hsv_features(image),                 # 18 features
        extract_texture_features(gray),              # 30 features
        extract_lbp_features(gray),                  # variable ~26
        extract_edge_features(gray),                 # 6 features
        extract_frequency_features(gray),            # 5 features
        extract_noise_features(gray),                # 5 features
        extract_saturation_uniformity(image),        # 5 features
    ])
    
    return features


def get_feature_names():
    """Return descriptive names for feature groups."""
    names = []
    for c in ['R', 'G', 'B']:
        names.extend([f'hist_{c}_{i}' for i in range(32)])
    for c in ['H', 'S', 'V']:
        names.extend([f'hsv_{c}_{s}' for s in ['mean','std','skew','kurt','q25','q75']])
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for p in props:
        names.extend([f'glcm_{p}_{i}' for i in range(6)])
    names.extend([f'lbp_{i}' for i in range(26)])
    names.extend(['edge_sobel_mean', 'edge_sobel_std', 'edge_canny_mean',
                  'edge_density', 'edge_sobel_p90', 'edge_sobel_p99'])
    names.extend([f'fft_ring_{i}' for i in range(4)] + ['fft_hf_ratio'])
    names.extend(['noise_mean', 'noise_std', 'noise_kurt', 'noise_skew', 'noise_p95'])
    names.extend(['sat_local_var_mean', 'sat_local_var_std', 'sat_mean', 'sat_std', 'sat_lv_p90'])
    return names

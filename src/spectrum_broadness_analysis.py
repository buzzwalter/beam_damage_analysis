import numpy as np
import cv2
from scipy import stats
import time
from functools import wraps
from src.common_imports import *
from src.common_imports import timer
from scipy.signal import savgol_filter
from scipy.stats import norm

def get_second_moment(fft_magnitude):
    center_y, center_x = np.array(fft_magnitude.shape) // 2
    y, x = np.ogrid[:fft_magnitude.shape[0], :fft_magnitude.shape[1]]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    distances = distances / np.max(distances)
    return np.sum(distances**2 * fft_magnitude) / np.sum(fft_magnitude)

def get_high_freq_power(fft_magnitude):
    center_y, center_x = np.array(fft_magnitude.shape) // 2
    y, x = np.ogrid[:fft_magnitude.shape[0], :fft_magnitude.shape[1]]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    max_distance = np.max(distances)
    threshold_radius = max_distance / 4

    # Calculate power ratio of high frequencies to total
    high_freq_mask = distances > threshold_radius
    high_freq_power = np.sum(fft_magnitude[high_freq_mask])
    total_power = np.sum(fft_magnitude)
    return high_freq_power / total_power

def get_radial_std(fft_magnitude, threshold_radius=50):
    # Get center coordinates
    center_y, center_x = np.array(fft_magnitude.shape) // 2
    y, x = np.ogrid[:fft_magnitude.shape[0], :fft_magnitude.shape[1]]

    # Calculate distances from center
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Subset data to exclude high frequencies
    high_freq_mask = distances < threshold_radius
    fft_magnitude = fft_magnitude[high_freq_mask]

    print(fft_magnitude.shape, distances.shape)
    # # Weight distances by magnitude
    # weighted_distances = distances * fft_magnitude
    # mean_distance = np.sum(weighted_distances) / np.sum(fft_magnitude)
    
    # # Calculate standard deviation
    # variance = np.sum(fft_magnitude * (distances - mean_distance)**2) / np.sum(fft_magnitude)
    # return np.sqrt(variance)
    distances = distances / np.max(distances)
    radial_bins = np.linspace(0, 1, 50)
    radial_profile = stats.binned_statistic(
        distances.ravel(),
        fft_magnitude.ravel(),
        statistic='mean',
        bins=radial_bins
    ).statistic
    
    # Remove NaN values that might occur in empty bins
    radial_profile = radial_profile[~np.isnan(radial_profile)]
    
    return np.std(radial_profile)



def analyze_spectrum_broadness(fft_spectrum):
    """
    Analyze the broadness of the frequency spectrum of an image
    
    Parameters:
    image: numpy array, input fft_spectrum
    
    Returns:
    dict: Various metrics of spectrum broadness
    """
    # # Convert to grayscale if needed
    # if len(image.shape) == 3:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # else:
    #     gray = image
    
    # # Compute FFT
    # f_transform = np.fft.fft2(gray.astype(float))
    # f_transform_shifted = np.fft.fftshift(f_transform)
    # magnitude = np.abs(f_transform_shifted)
    
    # # Calculate center
    # center_y, center_x = np.array(magnitude.shape) // 2
    # y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    # distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate metrics
    metrics = {}
    
    # 1. Second moment (fastest)
    
    second_moment, second_moment_duration = timer(get_second_moment)(fft_spectrum)
    metrics['second_moment'] = (second_moment, second_moment_duration*1000) #np.sum(distances**2 * magnitude) / np.sum(magnitude)
    
    # 2. High frequency power ratio (threshold at 1/4 of max distance)
    # max_distance = np.max(distances)
    # threshold = max_distance / 4
    # high_freq_mask = distances > threshold
    # metrics['high_freq_ratio'] = (np.sum(magnitude[high_freq_mask]) / 
    #                             np.sum(magnitude))
    high_freq_ratio, high_freq_duration = timer(get_high_freq_power)(fft_spectrum)
    metrics['high_freq_ratio'] = (high_freq_ratio, high_freq_duration*1000)

    # 3. standard deviation of radial distribution
    # radial_profile = stats.binned_statistic(
    #     distances.ravel(),
    #     magnitude.ravel(),
    #     statistic='mean',
    #     bins=50
    # ).statistic
    # metrics['radial_kurtosis'] = stats.kurtosis(radial_profile)
    radial_std, radial_std_duration = timer(get_radial_std)(fft_spectrum)
    metrics['radial_std'] = (radial_std, radial_std_duration*1000)

    return metrics

def compare_images(image1, image2):
    """
    Compare spectrum broadness between two images
    """
    metrics1 = analyze_spectrum_broadness(image1)
    metrics2 = analyze_spectrum_broadness(image2)
    
    print("Image 1 metrics:")
    for key, value in metrics1.items():
        print(f"{key}: {value[0]:.4f}, {value[1]:.4f} seconds")
    
    print("\nImage 2 metrics:")
    for key, value in metrics2.items():
        print(f"{key}: {value[0]:.4f}, {value[1]:.4f} seconds")
    
    # Calculate percent differences
    print("\nPercent differences:")
    for key in metrics1.keys():
        diff = ((metrics2[key][0] - metrics1[key][0]) / metrics1[key][0]) * 100
        print(f"{key}: {diff:.1f}%")
    
    return metrics1, metrics2

def get_first_derivative(radial_std_series, window_size=11, poly_order=3, 
                                threshold_sigma=2, smoothing_window=11):
    """
    Smooth the radial_std_series using Savitzky-Golay filter, then calculate the first derivative.
    
    Parameters:
    radial_std_series: np.array, time series of radial standard deviations
    window_size: int, window size for Savitzky-Golay filter
    poly_order: int, polynomial order for Savitzky-Golay filter
    threshold_sigma: float, number of standard deviations for threshold
    smoothing_window: int, window size for initial smoothing
    
    Returns:
    np.array, first derivative of the smoothed radial_std_series
    """
    # Smooth the data and calculate the first derivative
    smoothed_data = savgol_filter(radial_std_series, smoothing_window, 3)
    first_derivative = savgol_filter(smoothed_data, window_size, poly_order, deriv=1)   

    # Detect significant changes
    first_deriv_threshold = np.std(first_derivative) * threshold_sigma
    first_deriv_detection = np.abs(first_derivative) > first_deriv_threshold
    return first_derivative, first_deriv_detection, first_deriv_threshold, smoothed_data

def get_second_derivative(radial_std_series, window_size=11, poly_order=3, 
                                threshold_sigma=2, smoothing_window=11):
    """
    Smooth the radial_std_series using Savitzky-Golay filter, then calculate the second derivative.
    
    Parameters:
    radial_std_series: np.array, time series of radial standard deviations
    window_size: int, window size for Savitzky-Golay filter
    poly_order: int, polynomial order for Savitzky-Golay filter
    threshold_sigma: float, number of standard deviations for threshold
    smoothing_window: int, window size for initial smoothing
    
    Returns:
    np.array, second derivative of the smoothed radial_std_series
    """
    # Smooth the data and calculate the second derivative
    smoothed_data = savgol_filter(radial_std_series, smoothing_window, 3)
    second_derivative = savgol_filter(smoothed_data, window_size, poly_order, deriv=2)

    # Detect significant changes
    second_deriv_threshold = np.std(second_derivative) * threshold_sigma
    second_deriv_detection = np.abs(second_derivative) > second_deriv_threshold
    return second_derivative, second_deriv_detection, second_deriv_threshold, smoothed_data

def detect_damage_from_timeseries(radial_std_series, window_size=11, poly_order=3, 
                                threshold_sigma=2, smoothing_window=11):
    """
    Detect damage by analyzing derivatives of smoothed radial standard deviation time series.
    
    Parameters:
    radial_std_series: np.array, time series of radial standard deviations
    window_size: int, window size for Savitzky-Golay filter
    poly_order: int, polynomial order for Savitzky-Golay filter
    threshold_sigma: float, number of standard deviations for threshold
    smoothing_window: int, window size for initial smoothing
    
    Returns:
    dict containing detection results and analysis data
    """
    #

    # Obtain derivatives and indicators
    first_derivative, first_deriv_detection, first_deriv_threshold, smoothed_data = get_first_derivative(radial_std_series, window_size, poly_order, 
                                threshold_sigma, smoothing_window)
    second_derivative, second_deriv_detection, second_deriv_threshold, _ = get_second_derivative(radial_std_series, window_size, poly_order, 
                                threshold_sigma, smoothing_window)


    # Calculate metrics for both derivatives
    first_deriv_metrics = {
        'mean': np.mean(np.abs(first_derivative)),
        'std': np.std(first_derivative),
        'max': np.max(np.abs(first_derivative)),
        'threshold': first_deriv_threshold
    }
    
    second_deriv_metrics = {
        'mean': np.mean(np.abs(second_derivative)),
        'std': np.std(second_derivative),
        'max': np.max(np.abs(second_derivative)),
        'threshold': second_deriv_threshold
    }
    
    # Combined detection (can be adjusted based on requirements)
    damage_detected = np.any(first_deriv_detection) or np.any(second_deriv_detection)
    
    return {
        'damage_detected': damage_detected,
        'smoothed_data': smoothed_data,
        'first_derivative': first_derivative,
        'second_derivative': second_derivative,
        'first_deriv_detection': first_deriv_detection,
        'second_deriv_detection': second_deriv_detection,
        'first_deriv_metrics': first_deriv_metrics,
        'second_deriv_metrics': second_deriv_metrics
    }

def plot_detection_results(results, time_points=None):
    """
    Plot the detection results with derivatives and thresholds
    """
    import matplotlib.pyplot as plt
    
    if time_points is None:
        time_points = np.arange(len(results['smoothed_data']))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot smoothed data
    ax1.plot(time_points, results['smoothed_data'], label='Smoothed Data')
    ax1.set_title('Smoothed Radial STD')
    ax1.legend()
    
    # Plot first derivative
    ax2.plot(time_points, results['first_derivative'], label='First Derivative')
    ax2.axhline(y=results['first_deriv_metrics']['threshold'], color='r', linestyle='--', 
                label='Upper Threshold')
    ax2.axhline(y=-results['first_deriv_metrics']['threshold'], color='r', linestyle='--', 
                label='Lower Threshold')
    ax2.set_title('First Derivative')
    ax2.legend()
    
    # Plot second derivative
    ax3.plot(time_points, results['second_derivative'], label='Second Derivative')
    ax3.axhline(y=results['second_deriv_metrics']['threshold'], color='r', linestyle='--', 
                label='Upper Threshold')
    ax3.axhline(y=-results['second_deriv_metrics']['threshold'], color='r', linestyle='--', 
                label='Lower Threshold')
    ax3.set_title('Second Derivative')
    ax3.legend()
    
    plt.tight_layout()
    return fig

    # Example usage:
"""
# Assuming you have a time series of radial_std values:
radial_std_series = np.array([...])  # Your time series data

# Analyze the data
results = detect_damage_from_timeseries(radial_std_series)

# Print results
print(f"Damage detected: {results['damage_detected']}")
print("\nFirst Derivative Metrics:")
for key, value in results['first_deriv_metrics'].items():
    print(f"{key}: {value:.6f}")
print("\nSecond Derivative Metrics:")
for key, value in results['second_deriv_metrics'].items():
    print(f"{key}: {value:.6f}")

# Plot results
fig = plot_detection_results(results)
plt.show()
"""
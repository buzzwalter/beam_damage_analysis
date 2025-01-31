import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_diffraction_rings(image):
    """
    Analyze diffraction rings in an image using FFT
    
    Parameters:
    image: numpy array, 3-channel RGB/BGR image
    
    Returns:
    fig: plotly figure object with original, grayscale, and FFT visualization
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Convert to float and normalize
    gray_float = gray.astype(float) / 255.0
    
    # Apply Hanning window to reduce edge effects
    h = np.hanning(gray.shape[0])
    w = np.hanning(gray.shape[1])
    window = np.sqrt(np.outer(h, w))
    gray_windowed = gray_float * window
    
    # Compute 2D FFT
    f_transform = np.fft.fft2(gray_windowed)
    
    # Shift zero frequency to center
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Compute magnitude spectrum with enhanced contrast
    magnitude = np.abs(f_transform_shifted)
    # Apply power-law transformation to enhance faint features
    magnitude_enhanced = np.power(magnitude, 0.2)  # fifth root to enhance faint features
    # Normalize to [0,1] range
    magnitude_normalized = (magnitude_enhanced - magnitude_enhanced.min()) / (magnitude_enhanced.max() - magnitude_enhanced.min())
    # Scale to 0-255 for display
    
    # magnitude_spectrum = (magnitude_normalized * 255).astype(np.uint8)
    
    # # Create visualization
    # fig = make_subplots(rows=1, cols=3, 
    #                    subplot_titles=('Original', 'Grayscale', 'FFT Magnitude Spectrum'),
    #                    horizontal_spacing=0.05)
    
    # # Original image
    # fig.add_trace(
    #     go.Image(z=image),
    #     row=1, col=1
    # )
    
    # # Grayscale image
    # fig.add_trace(
    #     go.Image(z=gray), #, zmin=[0,0,0,0], zmax=[255,255,255,255]),
    #     row=1, col=2
    # )

    # # FFT magnitude spectrum
    # # spectrum_min = magnitude_spectrum.min()
    # # spectrum_max = magnitude_spectrum.max()
    # # print(f'Spectrum min: {spectrum_min}, max: {spectrum_max}')
    # fig.add_trace(
    #     go.Image(z=magnitude_spectrum),  #, zmin=[spectrum_min,spectrum_min,spectrum_min,spectrum_min], zmax=[spectrum_max,spectrum_max,spectrum_max,spectrum_max]),
    #     row=1, col=3
    # )
    
    # # Update layout
    # fig.update_layout(
    #     title='Diffraction Ring Analysis',
    #     width=1200,
    #     height=400,
    #     showlegend=False
    # )
    
    # # Print spectrum stats for debugging
    # print(f"Original spectrum range - Min: {magnitude.min():.4f}, Max: {magnitude.max():.4f}")
    # print(f"Enhanced spectrum range - Min: {magnitude_enhanced.min():.4f}, Max: {magnitude_enhanced.max():.4f}")
    
    # Create single plot
    fig = go.Figure(
        data=go.Heatmap(
            z=magnitude_normalized,
            colorscale='Viridis'
        )
    )
    
    fig.update_layout(
        title='FFT Magnitude Spectrum',
        width=600,
        height=600
    )
    
    print(f"Magnitude range: {magnitude.min():.4f} to {magnitude.max():.4f}")
    print(f"Enhanced range: {magnitude_normalized.min():.4f} to {magnitude_normalized.max():.4f}")

    return fig, magnitude_normalized

# Example usage for a single image
# Assuming image_container_recent_damage is a list/array of images
def analyze_multiple_images(image_container, start_index, end_index):
    """
    Analyze multiple images from a container
    
    Parameters:
    image_container: list/array of images
    """
    for i, img in enumerate(image_container[start_index:end_index]):
        print("Processing image ", start_index+i+1)
        fig, spectrum = analyze_diffraction_rings(img)
        fig.update_layout(title=f'Diffraction Ring Analysis - Image {i+1}')
        fig.show()
        
        # Optional: you can add additional analysis here, such as:
        # - Radial average of the FFT
        # - Peak detection in the spectrum
        # - Ring detection using Hough transform on the FFT

        # jupyter nbconvert --to html damage_analys.ipynb
        # jupyter nbconvert --to pdf damage_analys.ipynb  
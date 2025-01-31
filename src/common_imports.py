# common_imports.py

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Image processing
import cv2
import scipy.ndimage as ndimage
from scipy import stats

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
from scipy import signal
from scipy.fft import fft, fft2, ifft2, fftshift, ifftshift
from scipy.signal import savgol_filter
from scipy.stats import norm

# Utility
import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        # print(f"Function {func.__name__} took {duration:.4f} seconds")
        return result, duration
    return wrapper

# Optional: you can set common plotting styles here
# plt.style.use('seaborn')
# sns.set_style("whitegrid")

# Optional: set common numpy print options
# np.set_printoptions(precision=3, suppress=True)



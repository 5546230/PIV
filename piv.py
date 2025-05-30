import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from skimage.io import imread

path = 'alpha_15_100_SubSlidMinOverTime/B00001.tif'
# path = 'calibration/B00001.tif'
data = imread(path)

mid = data.shape[0]
pic_1 = data[:mid//2, :]
pic_2 = data[mid//2:, :]
print(pic_1.min(), pic_1.max(), pic_1.mean())
print(pic_2.min(), pic_2.max(), pic_2.mean())
# imporve the contrast

def enhance_contrast(image, method='adaptive_eq', **kwargs):
    """
    Enhance contrast of an image using various methods.
    
    Parameters:
    -----------
    image : ndarray
        Input image to enhance
    method : str
        Method to use for enhancement:
        - 'hist_eq': Simple histogram equalization
        - 'adaptive_eq': Contrast Limited Adaptive Histogram Equalization (CLAHE)
        - 'gamma': Gamma correction
        - 'stretch': Contrast stretching
        - 'sharp': Unsharp masking (sharpening)
    kwargs : dict
        Additional parameters for specific methods:
        - gamma: gamma value (default=0.5)
        - clip_limit: for adaptive_eq (default=2.0)
        - kernel_size: for adaptive_eq (default=8)
    
    Returns:
    --------
    enhanced : ndarray
        Enhanced image
    """
    import cv2
    from skimage import exposure
    from scipy import ndimage
    
    # Make sure image is in the right format and range
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
    
    if image.max() > 1.0:
        image = image / 255.0
        
    # Apply the selected enhancement method
    if method == 'hist_eq':
        # Simple histogram equalization
        enhanced = exposure.equalize_hist(image)
        
    elif method == 'adaptive_eq':
        # Contrast Limited Adaptive Histogram Equalization
        clip_limit = kwargs.get('clip_limit', 2.0)
        kernel_size = kwargs.get('kernel_size', 8)
        
        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kernel_size, kernel_size))
        enhanced = clahe.apply(img_uint8) / 255.0
        
    elif method == 'gamma':
        # Gamma correction
        gamma = kwargs.get('gamma', 0.5)  # gamma < 1 brightens, gamma > 1 darkens
        enhanced = exposure.adjust_gamma(image, gamma)
        
    elif method == 'stretch':
        # Contrast stretching
        p_low = kwargs.get('p_low', 2)
        p_high = kwargs.get('p_high', 98)
        enhanced = exposure.rescale_intensity(image, in_range=tuple(np.percentile(image, (p_low, p_high))))
        
    elif method == 'sharp':
        # Unsharp masking (sharpening)
        sigma = kwargs.get('sigma', 1.0)
        amount = kwargs.get('amount', 1.0)
        blurred = ndimage.gaussian_filter(image, sigma=sigma)
        enhanced = image + amount * (image - blurred)
        enhanced = np.clip(enhanced, 0, 1)
        
    else:
        raise ValueError(f"Unknown enhancement method: {method}")
        
    return enhanced

window_size = 32
def optimize_contrast_params(pic_1_orig, pic_2_orig, snr_threshold=1.2, max_iterations=10, 
                            window_size=31, verbose=True):
    """
    Optimize the contrast enhancement parameters to maximize the number of points with SNR > threshold.
    
    Parameters:
    -----------
    pic_1_orig, pic_2_orig : ndarray
        Original images to optimize
    snr_threshold : float
        SNR threshold to count as good points
    max_iterations : int
        Maximum number of iterations for optimization
    window_size : int
        Window size for PIV analysis
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    best_params : dict
        Dictionary with optimized parameters
    """
    from scipy.optimize import differential_evolution
    import time
    
    # Store original images
    orig_1 = pic_1_orig.copy()
    orig_2 = pic_2_orig.copy()
    
    def compute_snr_count(params):
        """Objective function for optimization: count points with SNR > threshold"""
        clip_limit_1, clip_limit_2, kernel_size_factor = params
        kernel_size = int(kernel_size_factor)  # Convert to integer
        
        # Reset images
        pic_1 = orig_1.copy()
        pic_2 = orig_2.copy()
        
        # Normalize images
        pic_1 = (pic_1 - np.min(pic_1)) / (np.max(pic_1) - np.min(pic_1))
        pic_2 = (pic_2 - np.min(pic_2)) / (np.max(pic_2) - np.min(pic_2))
        
        # Apply contrast enhancement with current parameters
        pic_1 = enhance_contrast(pic_1, method='adaptive_eq', 
                                clip_limit=clip_limit_1, 
                                kernel_size=kernel_size)
        pic_2 = enhance_contrast(pic_2, method='adaptive_eq', 
                                clip_limit=clip_limit_2, 
                                kernel_size=kernel_size)
        
        # Normalize again after enhancement
        pic_1 = (pic_1 - np.min(pic_1)) / (np.max(pic_1) - np.min(pic_1))
        pic_2 = (pic_2 - np.min(pic_2)) / (np.max(pic_2) - np.min(pic_2))
        
        # Collect SNR values
        snrs = []
        
        # Loop over a subset of image points to speed up optimization
        step = 5  # Skip some windows to make it faster
        for i in range(0, pic_1.shape[0] - window_size, window_size * step):
            for j in range(0, pic_1.shape[1] - window_size, window_size * step):
                # Extract the window from each image
                window1 = pic_1[i:i + window_size, j:j + window_size]
                
                # Determine search area in second image
                i2 = max(0, i - window_size)
                j2 = max(0, j - window_size)
                
                x_offset = window_size if i-window_size > 0 else i
                y_offset = window_size if j-window_size > 0 else j
                
                # Extract the window from the second image        
                window_2 = pic_2[i2:i + 2*window_size-1, j2:j + 2*window_size-1]
                
                # Compute the cross-correlation
                cor = correlate2d(window1, window_2, mode='valid')[::-1, ::-1]
                
                # Find the peak
                i_max, j_max = np.unravel_index(np.argmax(cor), cor.shape)
                max_val = cor[i_max, j_max]
                
                # Zero out peak neighborhood for SNR calculation
                cor_snr = cor.copy()
                cor_snr[max(0, i_max-4):min(cor.shape[0], i_max+5), 
                       max(0, j_max-4):min(cor.shape[1], j_max+5)] = 0
                
                # Calculate SNR
                SNR = max_val / np.max(cor_snr) if np.max(cor_snr) > 0 else max_val
                snrs.append(SNR)
        
        # Count points above threshold
        good_points = np.sum(np.array(snrs) > snr_threshold)
        
        if verbose:
            print(f"Testing: clip1={clip_limit_1:.1f}, clip2={clip_limit_2:.1f}, "
                 f"kernel={kernel_size}, good_points={good_points}/{len(snrs)}")
            
        # Return negative count for minimization
        return -good_points
    
    # Define bounds for parameters
    bounds = [(0.0, 40.0),     # clip_limit_1
              (0.0, 40.0),     # clip_limit_2
              (3, window_size)]  # kernel_size (as real number, will be cast to int)
    
    # Run optimization
    print(f"Starting optimization to maximize points with SNR > {snr_threshold}")
    print(f"This may take several minutes depending on image size...")
    start_time = time.time()
    
    result = differential_evolution(
        compute_snr_count,
        bounds,
        maxiter=max_iterations,
        popsize=10,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=verbose
    )
    
    # Extract best parameters
    best_clip1, best_clip2, best_kernel = result.x
    best_kernel = int(best_kernel)
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.1f} seconds")
    print(f"Best parameters: clip_limit_1={best_clip1:.1f}, "
         f"clip_limit_2={best_clip2:.1f}, kernel_size={best_kernel}")
    
    # Return the best parameters
    return {
        'clip_limit_1': best_clip1,
        'clip_limit_2': best_clip2,
        'kernel_size': best_kernel
    }

# Example usage:
# Store original images
pic_1_orig = data[:mid//2, :].copy()
pic_2_orig = data[mid//2:, :].copy()

# Uncomment these lines to run the optimizer
# best_params = optimize_contrast_params(pic_1_orig, pic_2_orig, snr_threshold=1.1)
# 
# # Apply the best parameters
# pic_1 = (pic_1_orig - np.min(pic_1_orig)) / (np.max(pic_1_orig) - np.min(pic_1_orig))
# pic_2 = (pic_2_orig - np.min(pic_2_orig)) / (np.max(pic_2_orig) - np.min(pic_2_orig))
# pic_1 = enhance_contrast(pic_1, method='adaptive_eq', clip_limit=best_params['clip_limit_1'], kernel_size=best_params['kernel_size'])
# pic_2 = enhance_contrast(pic_2, method='adaptive_eq', clip_limit=best_params['clip_limit_2'], kernel_size=best_params['kernel_size'])
# pic_1 = (pic_1 - np.min(pic_1)) / (np.max(pic_1) - np.min(pic_1))
# pic_2 = (pic_2 - np.min(pic_2)) / (np.max(pic_2) - np.min(pic_2))
# normalize the data
# pic_1 = (pic_1 - np.min(pic_1)) / (np.max(pic_1) - np.min(pic_1))
# pic_2 = (pic_2 - np.min(pic_2)) / (np.max(pic_2) - np.min(pic_2))
# # Example usage in your code:
# # Add this after loading and normalizing the images
# pic_1 = enhance_contrast(pic_1, method='adaptive_eq', clip_limit=30.0, kernel_size=window_size)
# pic_2 = enhance_contrast(pic_2, method='adaptive_eq', clip_limit=10.0, kernel_size=window_size)
# normalize the data
# pic_1 = (pic_1 - np.min(pic_1)) / (np.max(pic_1) - np.min(pic_1))
# pic_2 = (pic_2 - np.min(pic_2)) / (np.max(pic_2) - np.min(pic_2))
# pic_1 = enhance_contrast(pic_1, method='hist_eq')
# pic_2 = enhance_contrast(pic_2, method='hist_eq')
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(pic_1, cmap='gray')
axs[0].set_title('Image 1')
axs[1].imshow(pic_2, cmap='gray')
axs[1].set_title('Image 2')
plt.show()
# Alternatively, try other methods:
# pic_1 = enhance_contrast(pic_1, method='gamma', gamma=0.5)
# pic_1 = enhance_contrast(pic_1, method='stretch', p_low=2, p_high=98)

# Find velocity vectors through cross-correlation



l_px = 170e-3/1616 # length of a pixel in meters #TODO: check this value
dt = 70e-6 # time between images in seconds
# pic_1 = np.zeros_like(pic_1)
# pic_2 = np.zeros_like(pic_2)
# pic_1 = np.random.rand(*pic_1.shape)
# pic_2 = np.roll(pic_1, 3, axis=0)
# pic_2 = np.roll(pic_2, 4, axis=1)
# pic_1[0, :] = np.arange(0, pic_1.shape[1])
# pic_2[1, :] = np.arange(0, pic_2.shape[1])
def find_peak(cor):
    """
    Find the subpixel location of the peak in the cross-correlation using a Gaussian fit.
    """
    # cor = cor.T
    # Find the integer peak location
    i, j = np.unravel_index(np.argmax(cor), cor.shape)
    # return j, i

    # Ensure the peak is not on the border
    if i <= 0 or i >= cor.shape[0] - 1 or j <= 0 or j >= cor.shape[1] - 1:
        return j, i  # Return integer peak if on the border

    # Extract the neighboring values for Gaussian fitting
    ln_phi_i_minus_1_j = np.log(cor[i - 1, j])
    ln_phi_i_plus_1_j = np.log(cor[i + 1, j])
    ln_phi_i_j = np.log(cor[i, j])
    ln_phi_i_j_minus_1 = np.log(cor[i, j - 1])
    ln_phi_i_j_plus_1 = np.log(cor[i, j + 1])



    # Compute subpixel peak location using the Gaussian fit formula
    x_0 = j + (ln_phi_i_j_minus_1 - ln_phi_i_j_plus_1) / (
        2 * ln_phi_i_j_minus_1 - 4 * ln_phi_i_j + 2 * ln_phi_i_j_plus_1
    )
    y_0 = i + (ln_phi_i_minus_1_j - ln_phi_i_plus_1_j) / (
        2 * ln_phi_i_minus_1_j - 4 * ln_phi_i_j + 2 * ln_phi_i_plus_1_j
    )

    return x_0, y_0



def create_velocity_mask(image, mask_file='velocity_mask.npy'):
    """
    Create a mask by drawing polygons on regions where velocities should not be computed.
    
    Parameters:
    -----------
    image : ndarray
        The input image on which to draw the mask
    mask_file : str
        File path to save the mask
        
    Returns:
    --------
    mask : ndarray
        Binary mask where True indicates regions to exclude from velocity calculations
    """
    from matplotlib.widgets import PolygonSelector
    from matplotlib.path import Path
    
    # Initialize the mask
    mask = np.zeros_like(image, dtype=bool)
    
    # Create a figure for drawing the mask
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title('Draw polygons to mask regions (press Enter when done)')
    
    # Keep track of all polygons
    polygons = []
    
    def onselect(verts):
        # Create a polygon from vertices
        poly = np.array(verts)
        polygons.append(poly)
        
        # Draw the polygon
        polygon_patch = plt.Polygon(poly, fill=True, alpha=0.4, color='red')
        ax.add_patch(polygon_patch)
        fig.canvas.draw()
    
    # Create the polygon selector
    selector = PolygonSelector(ax, onselect)
    
    # Wait for user to press Enter
    plt.gcf().canvas.mpl_connect('key_press_event', 
                                 lambda event: plt.close() if event.key == 'enter' else None)
    plt.show()
    
    # Convert polygons to mask
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    points = np.vstack((x.ravel(), y.ravel())).T
    
    for poly in polygons:
        path = Path(poly)
        grid = path.contains_points(points).reshape(h, w)
        mask = mask | grid
    
    # Save the mask
    np.save(mask_file, mask)
    
    # Display the final mask
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(image, cmap='gray')
    ax[1].imshow(mask, alpha=0.5, cmap='jet')
    ax[1].set_title('Image with Mask')
    plt.tight_layout()
    plt.show()
    
    print(f"Mask saved to {mask_file}")
    return mask

def load_velocity_mask(mask_file='velocity_mask.npy', shape=None):
    """
    Load a velocity mask from file.
    If the file doesn't exist, return None or an empty mask.
    
    Parameters:
    -----------
    mask_file : str
        Path to the mask file
    shape : tuple, optional
        Shape for an empty mask if file doesn't exist
        
    Returns:
    --------
    mask : ndarray or None
        The loaded mask, or None if file doesn't exist and no shape provided
    """
    try:
        mask = np.load(mask_file)
        print(f"Loaded mask from {mask_file}")
        return mask
    except FileNotFoundError:
        if shape is not None:
            print(f"No mask found at {mask_file}, returning empty mask")
            return np.zeros(shape, dtype=bool)
        else:
            print(f"No mask found at {mask_file}")
            return None

# Example of how to use these functions in the main PIV code:
def apply_velocity_mask_to_piv(window_size=None, overlap=0):
    """
    Create or load a mask and downsample it to match the PIV grid dimensions.
    
    Parameters:
    -----------
    window_size : int, optional
        Size of the interrogation window, defaults to global window_size
    overlap : int
        Percentage of window overlap (0-99), e.g., 50 for 50% overlap
        
    Returns:
    --------
    mask : ndarray
        Full resolution mask
    mask_ds : ndarray
        Downsampled mask matching PIV grid dimensions
    """
    # Use global window_size if not provided
    if window_size is None:
        window_size = globals().get('window_size', 31)
    
    # Calculate step size based on overlap
    step = int(window_size * (1 - overlap/100))
    if step < 1:
        step = 1  # Ensure minimum step size
    
    # Load or create a mask
    mask_file = 'velocity_mask.npy'
    mask = load_velocity_mask(mask_file, pic_1.shape)
    
    if mask is None:
        # Let user create a mask
        mask = create_velocity_mask(pic_1, mask_file)
    
    # Calculate grid dimensions with overlap
    rows = len(range(0, pic_1.shape[0] - window_size + 1, step))
    cols = len(range(0, pic_1.shape[1] - window_size + 1, step))
    
    # Downsample the mask to match velocity grid dimensions with overlap
    mask_ds = np.zeros((rows, cols), dtype=bool)
    
    # Mark which velocity windows are masked
    for i_idx, i in enumerate(range(0, pic_1.shape[0] - window_size + 1, step)):
        for j_idx, j in enumerate(range(0, pic_1.shape[1] - window_size + 1, step)):
            # If any pixel in the window is masked, mask the whole window
            if np.any(mask[i:i+window_size, j:j+window_size]):
                mask_ds[i_idx, j_idx] = True
    
    print(f"Created mask with dimensions {mask_ds.shape} for PIV grid with {overlap}% overlap")
    return mask, mask_ds

# mask = create_velocity_mask(pic_2)
mask, mask_ds = apply_velocity_mask_to_piv(window_size, overlap=0)
print(window_size * l_px / dt*np.sqrt(2))
def cross_correlate_windows(window1, window2):
    """
    Compute the cross-correlation of two image windows.
    
    Parameters:
    -----------
    window1, window2 : ndarray
        Image windows to correlate
        
    Returns:
    --------
    cor : ndarray
        Cross-correlation result
    """
    # Compute the cross-correlation using FFT
    cor = correlate(window1 - window1.mean(), window2 - window2.mean(), method='fft')
    cor[cor<1e-8] = 1e-8
    return cor

def calculate_piv(pic_1, pic_2, window_size, l_px, dt, mask_ds=None, snr_threshold=1.07, overlap=0):
    """
    Calculate Particle Image Velocimetry (PIV) vectors from image pairs.
    
    Parameters:
    -----------
    pic_1, pic_2 : ndarray
        Input image pair for PIV analysis
    window_size : int
        Size of the interrogation window
    l_px : float
        Physical length of a pixel in meters
    dt : float
        Time between images in seconds
    mask_ds : ndarray, optional
        Downsampled mask where True indicates regions to exclude
    snr_threshold : float
        Signal-to-noise ratio threshold for valid vectors
    overlap : int
        Percentage of window overlap (0-99), e.g., 50 for 50% overlap
        
    Returns:
    --------
    U, V : ndarray
        Horizontal and vertical velocity components
    X, Y : ndarray
        Coordinates of velocity vectors
    snrs : ndarray
        Signal-to-noise ratios for each vector
    """
    # Calculate step size based on overlap
    step = int(window_size * (1 - overlap/100))
    if step < 1:
        step = 1  # Ensure minimum step size
    
    # Calculate grid dimensions with overlap
    rows = len(range(0, pic_1.shape[0] - window_size + 1, step))
    cols = len(range(0, pic_1.shape[1] - window_size + 1, step))
    
    # Create result arrays
    V = np.zeros((rows, cols))
    U = np.zeros((rows, cols))
    snrs = np.zeros((rows, cols))
    
    # Create grid coordinates with overlap
    X, Y = np.meshgrid(
        np.arange(0, pic_1.shape[1] - window_size + 1, step),
        np.arange(0, pic_1.shape[0] - window_size + 1, step)
    )
    
    # Check if mask needs resizing for overlap
    if mask_ds is not None and (mask_ds.shape[0] != rows or mask_ds.shape[1] != cols):
        print(f"Warning: Mask dimensions ({mask_ds.shape}) don't match velocity grid ({rows}, {cols})")
        print("Consider recreating the mask with the same overlap settings")
        # Create a new mask with correct dimensions (simple approach)
        new_mask = np.zeros((rows, cols), dtype=bool)
        # Copy values where possible (this is a simple approach and may not be optimal)
        min_rows = min(mask_ds.shape[0], rows)
        min_cols = min(mask_ds.shape[1], cols)
        new_mask[:min_rows, :min_cols] = mask_ds[:min_rows, :min_cols]
        mask_ds = new_mask
    
    # Loop over the image in windows with overlap
    for i_idx, i in enumerate(range(0, pic_1.shape[0] - window_size + 1, step)):
        for j_idx, j in enumerate(range(0, pic_1.shape[1] - window_size + 1, step)):
            # Skip if masked
            if mask_ds is not None and i_idx < mask_ds.shape[0] and j_idx < mask_ds.shape[1] and mask_ds[i_idx, j_idx]:
                U[i_idx, j_idx] = np.nan
                V[i_idx, j_idx] = np.nan
                continue
                
            # Extract the window from first image
            window1 = pic_1[i:i + window_size, j:j + window_size]
            # Extract the window from the second image
            window2 = pic_2[i:i + window_size, j:j + window_size]
    
            
            # Compute the cross-correlation with normalization
            try:
                cor = cross_correlate_windows(window1, window2)
                # fig, ax = plt.subplots()
                # ax.imshow(cor, cmap='gray')

            except Exception as e:
                print(f"Warning: Correlation failed at ({i},{j}): {e}")
                U[i_idx, j_idx] = np.nan
                V[i_idx, j_idx] = np.nan
                continue
            
            # Find the peak in the cross-correlation
            x_peak, y_peak = find_peak(cor)
            # plt.scatter(x_peak, y_peak, color='red')
            # plt.show()
            # x_peak = x_peak - x_offset
            # y_peak = y_peak - y_offset
            
            i_max, j_max = np.unravel_index(np.argmax(cor), cor.shape)
            max_val = cor[i_max, j_max]
            
            # Zero out area around peak for SNR calculation
            cor_snr = cor.copy()
            
            # Make sure indices stay within bounds
            i_min = max(0, i_max-4)
            i_max_bound = min(cor.shape[0], i_max+5)
            j_min = max(0, j_max-4)
            j_max_bound = min(cor.shape[1], j_max+5)
            
            cor_snr[i_min:i_max_bound, j_min:j_max_bound] = 0
            
            # Calculate SNR
            second_peak = np.max(cor_snr)
            SNR = max_val / second_peak if second_peak > 0 else max_val
            snrs[i_idx, j_idx] = SNR
            # print(x_peak, y_peak)
            # if x_peak > window_size/2:
            x_peak -= window_size - 1
            # if y_peak > window_size/2:
            y_peak -= window_size - 1
            # x_peak = x_peak# - window_size/2
            # y_peak = y_peak# - window_size/2
            
            # Apply SNR threshold
            if SNR < snr_threshold:
                u, v = np.nan, np.nan
            else:
                # Compute the velocity vector
                u = (-x_peak * l_px) / dt
                v = (y_peak * l_px) / dt  # Negative for y since image coordinates increase downward
            
            # Store the velocity vector
            V[i_idx, j_idx] = v
            U[i_idx, j_idx] = u
    
    # Detect outliers in U, V based on their neighbors
    # for i_idx in range(1, rows - 1):
    #     for j_idx in range(1, cols - 1):
    #         if mask_ds is not None and mask_ds[i_idx, j_idx]:
    #             continue
    #         # Get the 10x10 neighborhood
    #         u_neigh = U[i_idx-1:i_idx+2, j_idx-1:j_idx+2].flatten()
    #         v_neigh = V[i_idx-1:i_idx+2, j_idx-1:j_idx+2].flatten()

    #         # Calculate the median and standard deviation
    #         u_median = np.nanmedian(u_neigh)
    #         v_median = np.nanmedian(v_neigh)
    #         u_std = np.nanstd(u_neigh)
    #         v_std = np.nanstd(v_neigh)
    #         # Check if the current value is an outlier
    #         if np.abs(U[i_idx, j_idx] - u_median) > 3 * u_std or np.abs(V[i_idx, j_idx] - v_median) > 3 * v_std:
    #             U[i_idx, j_idx] = u_median
    #             V[i_idx, j_idx] = v_median
    #             # snrs[i_idx, j_idx] = np.nan


    return U, V, X, Y, snrs


def calculate_piv_multipass(pic_1, pic_2, window_size, l_px, dt, passes, mask_ds=None, snr_threshold=1.07, overlap=0):
    """
    Calculate Particle Image Velocimetry (PIV) vectors from image pairs using multipass method.
    
    Parameters:
    -----------
    pic_1, pic_2 : ndarray
        Input image pair for PIV analysis
    window_size : int
        Size of the initial interrogation window
    l_px : float
        Physical length of a pixel in meters
    dt : float
        Time between images in seconds
    passes : int
        Number of passes for multipass PIV analysis
    mask_ds : ndarray, optional
        Downsampled mask where True indicates regions to exclude
    snr_threshold : float
        Signal-to-noise ratio threshold for valid vectors
    overlap : int
        Percentage of window overlap (0-99), e.g., 50 for 50% overlap
        
    Returns:
    --------
    U, V : ndarray
        Horizontal and vertical velocity components (in physical units, m/s)
    X, Y : ndarray
        Coordinates of velocity vectors (in pixels)
    snrs : ndarray
        Signal-to-noise ratios for each vector (from the last successful pass)
    """
    # Calculate step size based on overlap
    step = int(window_size * (1 - overlap/100))
    if step < 1:
        step = 1  # Ensure minimum step size
    
    # Calculate grid dimensions with overlap
    rows = len(range(0, pic_1.shape[0] - window_size + 1, step))
    cols = len(range(0, pic_1.shape[1] - window_size + 1, step))
    
    # Create result arrays
    V_pix = np.zeros((rows, cols)) # Store pixel displacements
    U_pix = np.zeros((rows, cols)) # Store pixel displacements
    snrs_map = np.zeros((rows, cols))
    
    # Create grid coordinates with overlap
    X, Y = np.meshgrid(
        np.arange(0, pic_1.shape[1] - window_size + 1, step),
        np.arange(0, pic_1.shape[0] - window_size + 1, step)
    )
    
    # Check if mask needs resizing for overlap
    if mask_ds is not None and (mask_ds.shape[0] != rows or mask_ds.shape[1] != cols):
        print(f"Warning: Mask dimensions ({mask_ds.shape}) don't match velocity grid ({rows}, {cols})")
        print("Consider recreating the mask with the same overlap settings")
        new_mask = np.zeros((rows, cols), dtype=bool)
        min_r, min_c = min(mask_ds.shape[0], rows), min(mask_ds.shape[1], cols)
        new_mask[:min_r, :min_c] = mask_ds[:min_r, :min_c]
        mask_ds = new_mask
    
    # Loop over the image in windows with overlap
    for i_idx, i_start in enumerate(range(0, pic_1.shape[0] - window_size + 1, step)):
        for j_idx, j_start in enumerate(range(0, pic_1.shape[1] - window_size + 1, step)):
            # Skip if masked
            if mask_ds is not None and i_idx < mask_ds.shape[0] and j_idx < mask_ds.shape[1] and mask_ds[i_idx, j_idx]:
                U_pix[i_idx, j_idx] = np.nan
                V_pix[i_idx, j_idx] = np.nan
                snrs_map[i_idx, j_idx] = np.nan
                continue
            
            # Extract the initial full-size interrogation windows
            window1_orig = pic_1[i_start : i_start + window_size, j_start : j_start + window_size]
            window2_orig = pic_2[i_start : i_start + window_size, j_start : j_start + window_size]

            total_dx = 0.0
            total_dy = 0.0
            
            # Initialize U,V to NaN. They will be updated if any pass is successful.
            U_pix[i_idx, j_idx] = np.nan
            V_pix[i_idx, j_idx] = np.nan

            for pass_num in range(passes):
                current_pass_wind_size = window_size // (2**pass_num)
                if current_pass_wind_size < 4: # Minimum practical window size
                    break

                # Determine sub-window centers based on accumulated displacement (DWO)
                offset_x = total_dx / 2.0
                offset_y = total_dy / 2.0

                center1_x_in_orig = window_size / 2.0 + offset_x
                center1_y_in_orig = window_size / 2.0 + offset_y
                center2_x_in_orig = window_size / 2.0 - offset_x
                center2_y_in_orig = window_size / 2.0 - offset_y

                # Top-left of sub-window1 in window1_orig
                tl1_x = int(round(center1_x_in_orig - current_pass_wind_size / 2.0))
                tl1_y = int(round(center1_y_in_orig - current_pass_wind_size / 2.0))
                
                # Top-left of sub-window2 in window2_orig
                tl2_x = int(round(center2_x_in_orig - current_pass_wind_size / 2.0))
                tl2_y = int(round(center2_y_in_orig - current_pass_wind_size / 2.0))

                # Boundary checks for sub-window extraction
                tl1_x = max(0, min(tl1_x, window_size - current_pass_wind_size))
                tl1_y = max(0, min(tl1_y, window_size - current_pass_wind_size))
                tl2_x = max(0, min(tl2_x, window_size - current_pass_wind_size))
                tl2_y = max(0, min(tl2_y, window_size - current_pass_wind_size))

                sub_window1 = window1_orig[tl1_y : tl1_y + current_pass_wind_size, tl1_x : tl1_x + current_pass_wind_size]
                sub_window2 = window2_orig[tl2_y : tl2_y + current_pass_wind_size, tl2_x : tl2_x + current_pass_wind_size]
                
                if sub_window1.shape[0] < 4 or sub_window1.shape[1] < 4 or \
                   sub_window2.shape[0] < 4 or sub_window2.shape[1] < 4 or \
                   sub_window1.size == 0 or sub_window2.size == 0:
                    if pass_num == 0: # If first pass fails due to window issues, mark NaN
                        U_pix[i_idx, j_idx] = np.nan
                        V_pix[i_idx, j_idx] = np.nan
                    break 

                try:
                    cor = cross_correlate_windows(sub_window1, sub_window2)
                except Exception:
                    if pass_num == 0:
                        U_pix[i_idx, j_idx] = np.nan
                        V_pix[i_idx, j_idx] = np.nan
                    break 
                
                x_peak_cor, y_peak_cor = find_peak(cor) 
                
                dx_current = x_peak_cor - (current_pass_wind_size - 1)
                dy_current = y_peak_cor - (current_pass_wind_size - 1)
                
                i_max_cor, j_max_cor = np.unravel_index(np.argmax(cor), cor.shape)
                max_val = cor[i_max_cor, j_max_cor]
                cor_snr_calc = cor.copy()
                i_min_snr = max(0, i_max_cor - 4)
                i_max_bound_snr = min(cor.shape[0], i_max_cor + 5)
                j_min_snr = max(0, j_max_cor - 4)
                j_max_bound_snr = min(cor.shape[1], j_max_cor + 5)
                
                if i_min_snr < i_max_bound_snr and j_min_snr < j_max_bound_snr: # Ensure valid slice
                    cor_snr_calc[i_min_snr:i_max_bound_snr, j_min_snr:j_max_bound_snr] = 0
                
                second_peak = np.max(cor_snr_calc)
                current_SNR = max_val / second_peak if second_peak > 1e-9 else max_val # Avoid division by zero/small
                
                snrs_map[i_idx, j_idx] = current_SNR # Store SNR of the current pass

                if current_SNR < snr_threshold:
                    if pass_num == 0: 
                        U_pix[i_idx, j_idx] = np.nan 
                        V_pix[i_idx, j_idx] = np.nan
                    break 
                
                total_dx += dx_current
                total_dy += dy_current
                
                # If this pass was successful, update U_pix and V_pix with current total
                U_pix[i_idx, j_idx] = total_dx 
                V_pix[i_idx, j_idx] = total_dy


    # Outlier detection and replacement (applied to pixel displacements)
    for _ in range(3): # Iterative outlier removal
        U_temp = U_pix.copy()
        V_temp = V_pix.copy()
        for i_r in range(rows):
            for j_c in range(cols):
                if mask_ds is not None and i_r < mask_ds.shape[0] and j_c < mask_ds.shape[1] and mask_ds[i_r, j_c]:
                    continue
                if np.isnan(U_pix[i_r, j_c]):
                    continue

                # Define neighborhood (e.g., 3x3 or 5x5, excluding NaNs and masked points)
                u_neigh, v_neigh = [], []
                # Using a 3x3 neighborhood for simplicity here. For larger, adjust bounds.
                for ni in range(max(0, i_r-1), min(rows, i_r+2)):
                    for nj in range(max(0, j_c-1), min(cols, j_c+2)):
                        if (ni == i_r and nj == j_c):
                            continue
                        if mask_ds is not None and ni < mask_ds.shape[0] and nj < mask_ds.shape[1] and mask_ds[ni, nj]:
                            continue
                        if not np.isnan(U_temp[ni, nj]): # Use U_temp for consistent comparison within iteration
                            u_neigh.append(U_temp[ni, nj])
                            v_neigh.append(V_temp[ni, nj])
                
                if not u_neigh: # No valid neighbors
                    continue

                u_median = np.median(u_neigh)
                v_median = np.median(v_neigh)
                
                # Normalized median test (from Westerweel & Scarano 2005)
                # Fluctuations of u relative to median of neighbors
                u_fluct = [u - u_median for u in u_neigh]
                v_fluct = [v - v_median for v in v_neigh]
                
                median_u_fluct = np.median(np.abs(u_fluct))
                median_v_fluct = np.median(np.abs(v_fluct))

                # Threshold (e.g., 2 times the median fluctuation)
                # Using a simpler std-dev based approach from original code for now
                u_std = np.std(u_neigh) if len(u_neigh) > 1 else 0
                v_std = np.std(v_neigh) if len(v_neigh) > 1 else 0

                # If std is very small, avoid overly aggressive filtering
                u_std = max(u_std, 1e-2) # Minimum std deviation to consider
                v_std = max(v_std, 1e-2)


                if np.abs(U_pix[i_r, j_c] - u_median) > 2 * u_std or \
                   np.abs(V_pix[i_r, j_c] - v_median) > 2 * v_std:
                    U_pix[i_r, j_c] = u_median # Replace with median
                    V_pix[i_r, j_c] = v_median
                    # Or, could set to NaN if preferred:
                    # U_pix[i_r, j_c] = np.nan
                    # V_pix[i_r, j_c] = np.nan


    # Convert pixel displacements to physical velocities
    U = -U_pix * l_px / dt
    V = V_pix * l_px / dt # Negative sign for V if y-axis of image is downward and physical V is upward

    return U, V, X, Y, snrs_map

#6, 
snr_threshold = 1.2
reference_data = np.loadtxt('data\\alpha_15_100_PIV_MP(3x32x32_50ov)=unknown\\B00006.dat', skiprows=3)
refX = reference_data[:, 0]
refY = reference_data[:, 1]
l_px = (refX.max() - refX.min()) / (pic_1.shape[1] ) * 1e-3
print(f'{l_px=}')
dt = 70e-6
# a1 = np.array(((0, 1), (2, 0)))
# print(a1, np.roll(a1, 1, axis=0))
# pic_1 = np.random.rand(*pic_1.shape)
# pic_2 = np.roll(pic_1, 4, axis=0)
# pic_2 = np.roll(pic_2, 3, axis=1)
window_size = 32
# Example usage (replace the manual PIV computation with function call):
mask, mask_ds = apply_velocity_mask_to_piv(window_size, 50)
# U, V, X, Y, snrs = calculate_piv(pic_1, pic_2, window_size, l_px, dt, mask_ds, snr_threshold, 50)
U, V, X, Y, snrs = calculate_piv_multipass(pic_1, pic_2, window_size, l_px, dt,3, mask_ds, snr_threshold, 50)
U[mask_ds==1], V[mask_ds==1] = np.nan, np.nan
mask_1 = mask
# print(U[~np.isnan(U)].max())




def visualize_piv_results(U, V, X, Y, snrs=None, mask=None, pic_background=None, 
                          plot_type='all', save_path=None, figsize=(15, 10),
                          vector_scale=500, vector_width=0.002, vector_spacing=1,
                          cmap='jet', snr_threshold=1.0, dpi=300):
    """
    Visualize PIV analysis results with customizable options.
    
    Parameters:
    -----------
    U, V : ndarray
        Horizontal and vertical velocity components
    X, Y : ndarray
        Coordinates of velocity vectors
    snrs : ndarray, optional
        Signal-to-noise ratios for vectors
    mask : ndarray, optional
        Mask where True values are excluded from visualization
    pic_background : ndarray, optional
        Original image to use as background for vectors
    plot_type : str
        Type of plot: 'vectors', 'magnitude', 'components', 'snr', or 'all'
    save_path : str, optional
        Path to save the figure (if None, displays but doesn't save)
    figsize : tuple
        Figure size in inches
    vector_scale : float
        Scaling factor for vector arrows
    vector_width : float
        Width of vector arrows
    vector_spacing : int
        Display every nth vector (to avoid overcrowding)
    cmap : str
        Colormap for velocity magnitude and SNR plots
    snr_threshold : float
        Threshold for displaying SNR values
    dpi : int
        DPI for saved figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np
    
    # Apply mask if provided
    if mask is not None:
        print(U.shape, mask.shape)
        U_masked = np.copy(U)
        V_masked = np.copy(V)
        U_masked[mask==1] = np.nan
        V_masked[mask==1] = np.nan
    else:
        U_masked = U
        V_masked = V
    
    # Calculate velocity magnitude
    U_magnitude = np.sqrt(U_masked**2 + V_masked**2)
    
    # Set up figure based on plot_type
    if plot_type == 'vectors':
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[0]))
        axes = [ax]
    elif plot_type == 'magnitude':
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[0]))
        axes = [ax]
    elif plot_type == 'components':
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    elif plot_type == 'contour_vectors':
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[0]))
        axes = [ax]
    elif plot_type == 'snr' and snrs is not None:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[0]))
        axes = [ax]
    elif plot_type == 'all':
        if snrs is not None:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            axes = axes.flatten()
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")
    
    # Plot based on type
    plot_index = 0
    
    # Background image
    if pic_background is not None and plot_type in ['vectors', 'contour_vectors', 'all']:
        axes[plot_index].imshow(pic_background, cmap='gray', alpha=0.3)
    
    # Vectors plot
    if plot_type in ['vectors', 'all']:
        # Subsample vectors for cleaner display
        X_sub = X[::vector_spacing, ::vector_spacing]
        Y_sub = Y[::vector_spacing, ::vector_spacing]
        U_sub = U_masked[::vector_spacing, ::vector_spacing]
        V_sub = V_masked[::vector_spacing, ::vector_spacing]
        
        axes[plot_index].quiver(X_sub, Y_sub, U_sub, V_sub, 
                               color='k', scale=vector_scale, width=vector_width)
        
        axes[plot_index].set_title('Velocity Vectors')
        axes[plot_index].set_xlabel('X position (pixels)')
        axes[plot_index].set_ylabel('Y position (pixels)')
        plot_index += 1
    
    # Velocity magnitude
    if plot_type in ['magnitude', 'all']:
        # Create a colormap with NaN values as transparent
        cmap_with_alpha = plt.cm.get_cmap(cmap).copy()
        cmap_with_alpha.set_bad(alpha=0)
        
        # Color mapping
        norm = Normalize(vmin=np.nanmin(U_magnitude), vmax=np.nanmax(U_magnitude))
        
        # Plot the magnitude as a colored contour
        c = axes[plot_index].pcolormesh(X, Y, U_magnitude, cmap=cmap_with_alpha, 
                                       shading='auto', norm=norm)
        cbar = plt.colorbar(c, ax=axes[plot_index])
        cbar.set_label('Velocity Magnitude (m/s)')
        
        axes[plot_index].set_title('Velocity Magnitude')
        axes[plot_index].set_xlabel('X position (pixels)')
        axes[plot_index].set_ylabel('Y position (pixels)')
        plot_index += 1
    
    # Components (U and V separately)
    if plot_type in ['components', 'all']:
        # U component
        c = axes[plot_index].pcolormesh(X, Y, U_masked, cmap=cmap, shading='auto')
        cbar = plt.colorbar(c, ax=axes[plot_index])
        cbar.set_label('U Velocity (m/s)')
        axes[plot_index].set_title('U Velocity Component')
        axes[plot_index].set_xlabel('X position (pixels)')
        axes[plot_index].set_ylabel('Y position (pixels)')
        plot_index += 1
        
        # V component
        c = axes[plot_index].pcolormesh(X, Y, V_masked, cmap=cmap, shading='auto')
        cbar = plt.colorbar(c, ax=axes[plot_index])
        cbar.set_label('V Velocity (m/s)')
        axes[plot_index].set_title('V Velocity Component')
        axes[plot_index].set_xlabel('X position (pixels)')
        axes[plot_index].set_ylabel('Y position (pixels)')
        plot_index += 1
    
    # SNR plot
    if snrs is not None and plot_type in ['snr', 'all']:
        # Threshold SNR values
        snrs_viz = snrs.copy()
        snrs_viz[snrs_viz < snr_threshold] = np.nan
        
        # Plot SNR
        c = axes[plot_index].pcolormesh(X, Y, snrs_viz, cmap=cmap, shading='auto')
        cbar = plt.colorbar(c, ax=axes[plot_index])
        cbar.set_label('Signal-to-Noise Ratio')
        axes[plot_index].set_title(f'SNR (threshold={snr_threshold})')
        axes[plot_index].set_xlabel('X position (pixels)')
        axes[plot_index].set_ylabel('Y position (pixels)')
    
    # Add a new section for contour plot with vectors
    if plot_type in ['contour_vectors', 'all']:
        from scipy import interpolate
        
        # Create a more refined grid for smoother contours
        upscale_factor = 4  # Increase resolution by this factor
        x_fine = np.linspace(np.nanmin(X), np.nanmax(X), X.shape[1] * upscale_factor)
        y_fine = np.linspace(np.nanmin(Y), np.nanmax(Y), Y.shape[0] * upscale_factor)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        fine_mask = np.repeat(mask, upscale_factor, axis=0)
        fine_mask = np.repeat(fine_mask, upscale_factor, axis=1)
        # Get valid points for interpolation
        valid_mask = ~np.isnan(U_magnitude)
        # X_fine, Y_fine = X_fine[fine_mask==1], Y_fine[fine_mask==1]
        
        if np.sum(valid_mask) > 4:  # Need at least some valid points
            # Prepare points for interpolation
            x_valid = X[valid_mask]
            y_valid = Y[valid_mask]
            z_valid = U_magnitude[valid_mask]
            
            try:
                # Interpolate using griddata
                U_magnitude_interp = interpolate.griddata(
                    (x_valid, y_valid), z_valid,
                    (X_fine, Y_fine), method='cubic'
                )
                # print(U_magnitude_interp.shape)
                U_magnitude_interp[fine_mask==1] = np.nan
                # Apply the same colormap with transparency for NaN values
                cmap_with_alpha = plt.cm.get_cmap(cmap).copy()
                cmap_with_alpha.set_bad(alpha=0)
                
                # Create contour plot
                levels = np.linspace(np.nanmin(U_magnitude), np.nanmax(U_magnitude), 10)
                contour = axes[plot_index].contourf(
                    X_fine, Y_fine, U_magnitude_interp, 
                    levels=10, cmap=cmap_with_alpha, alpha=0.8
                )
                
                # Add color bar
                cbar = plt.colorbar(contour, ax=axes[plot_index])
                cbar.set_label('Velocity Magnitude (m/s)')
                
                # Overlay vectors
                X_sub = X[::vector_spacing, ::vector_spacing]
                Y_sub = Y[::vector_spacing, ::vector_spacing]
                U_sub = U_masked[::vector_spacing, ::vector_spacing]
                V_sub = V_masked[::vector_spacing, ::vector_spacing]
                
                # axes[plot_index].quiver(X_sub, Y_sub, U_sub, V_sub, 
                #                        color='k', scale=vector_scale, width=vector_width)
                # axes[plot_index].streamplot(
                #     X_sub, Y_sub, U_sub, V_sub, 
                #     color='k', linewidth=0.5, density=10, arrowsize=0.8
                # )
                
                # Add streamlines for flow visualization
                if np.sum(~np.isnan(U_masked)) > 10 and np.sum(~np.isnan(V_masked)) > 10:
                    axes[plot_index].streamplot(
                        X, Y, U_masked, V_masked, 
                        color='k', linewidth=0.5, density=1, arrowsize=0.8
                    )
                
                axes[plot_index].set_title('Velocity Field (Spline Interpolated)')
                axes[plot_index].set_xlabel('X position (pixels)')
                axes[plot_index].set_ylabel('Y position (pixels)')
            except Exception as e:
                axes[plot_index].text(0.5, 0.5, f"Interpolation failed: {e}", 
                                    ha='center', va='center', transform=axes[plot_index].transAxes)
        else:
            axes[plot_index].text(0.5, 0.5, "Not enough valid data points for interpolation", 
                                ha='center', va='center', transform=axes[plot_index].transAxes)
        
        plot_index += 1
    
    # Set appropriate aspect ratio and limits for all subplots
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(np.nanmin(X), np.nanmax(X))
        ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    return fig

# Example usage:
# U = U[1:-1, 1:-1]
# V = V[1:-1, 1:-1]
# X = X[1:-1, 1:-1]
# Y = Y[1:-1, 1:-1]
# snrs = snrs[1:-1, 1:-1]
# mask_ds = mask_ds[1:-1, 1:-1]
# visualize_piv_results(U, V, X, Y, snrs, mask_ds, plot_type='contour_vectors', save_path='piv_contour.png')
# visualize_piv_results(U, np.zeros_like(V), X, Y, snrs, mask_ds, plot_type='contour_vectors', save_path='piv_contour_u.png')
# visualize_piv_results(np.zeros_like(U), V, X, Y, snrs, mask_ds, plot_type='contour_vectors', save_path='piv_contour_v.png')
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage

# Function to apply Gaussian filter while ignoring NaN values
def nanfilter(U, sigma=1):
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = gaussian_filter(V, sigma=sigma)
    
    W = np.ones_like(U)
    W[np.isnan(U)] = 0
    WW = gaussian_filter(W, sigma=sigma)
    
    # Avoid division by zero
    WW[WW < 1e-10] = 1
    
    Z = VV / WW
    # Restore NaN values where they were originally
    Z[np.isnan(U)] = np.nan
    return Z

def sigma_from_filter(sigma):
    return sigma / np.sqrt(12)


# Helper function to calculate 1D Autocorrelation Function (ACF)
def _calculate_acf_1d(series, nlags):
    """Calculates ACF for a 1D series, handling NaNs."""
    mean_val = np.nanmean(series)
    centered_series = series - mean_val
    variance = np.nanvar(series)

    # Check if variance is zero or not enough data points
    if variance == 0 or np.sum(~np.isnan(series)) < 2:
        return np.full(nlags + 1, np.nan)

    acf_values = np.full(nlags + 1, np.nan)
    acf_values[0] = 1.0  # ACF at lag 0 is 1

    for lag in range(1, nlags + 1):
        if lag >= len(centered_series): # Ensure lag is within bounds
            break
        
        product_sum = 0
        pair_count = 0
        # Sum products of (value_i * value_{i+lag}) for available pairs
        for i in range(len(centered_series) - lag):
            val1 = centered_series[i]
            val2 = centered_series[i + lag]
            if not (np.isnan(val1) or np.isnan(val2)): # Check if both values are non-NaN
                product_sum += val1 * val2
                pair_count += 1
        
        if pair_count > 0: # If valid pairs found for this lag
            acf_values[lag] = (product_sum / pair_count) / variance
        else:
            acf_values[lag] = np.nan # No valid pairs, ACF is undefined for this lag
    return acf_values

# Function to estimate correlation lengths theta_1 and theta_2
def estimate_correlation_lengths_from_field(field_component, X_coords, Y_coords):
    """
    Estimates correlation lengths theta_1 (x-dir) and theta_2 (y-dir) for a 2D field.
    field_component: 2D numpy array (e.g., U or V velocity component).
    X_coords, Y_coords: 2D meshgrid of coordinates (typically in pixels).
    Returns: Tuple (theta_1, theta_2) in the same units as X_coords/Y_coords.
    """
    
    # Estimate grid spacing, assuming a regular grid
    dx = 1.0 # Default spacing
    if X_coords.shape[1] > 1:
        dx = np.abs(X_coords[0, 1] - X_coords[0, 0])
    
    dy = 1.0 # Default spacing
    if Y_coords.shape[0] > 1:
        dy = np.abs(Y_coords[1, 0] - Y_coords[0, 0])

    n_rows, n_cols = field_component.shape
    
    # --- Estimate theta_1 (correlation length in x-direction) ---
    max_lag_x = n_cols // 2
    if max_lag_x < 1: max_lag_x = 1 # Ensure at least one lag is possible
    
    all_acfs_x = []
    for i in range(n_rows): # Iterate over each row
        row_data = field_component[i, :]
        if np.sum(~np.isnan(row_data)) > 1: # Check for sufficient non-NaN data
            acf_x = _calculate_acf_1d(row_data, nlags=max_lag_x)
            all_acfs_x.append(acf_x)
    
    avg_acf_x = np.full(max_lag_x + 1, np.nan) # Initialize with NaNs
    if all_acfs_x: # If any ACFs were computed
        avg_acf_x = np.nanmean(np.array(all_acfs_x), axis=0)

    theta_1_pixels = np.nan
    # Find lag where ACF drops below 1/e, with interpolation
    for lag in range(1, len(avg_acf_x)):
        if not np.isnan(avg_acf_x[lag]) and avg_acf_x[lag] < (1/np.e):
            if not np.isnan(avg_acf_x[lag-1]) and avg_acf_x[lag-1] > (1/np.e) and \
               (avg_acf_x[lag-1] - avg_acf_x[lag] > 1e-9): # Check for valid interpolation
                 theta_1_pixels = (lag - 1) + (avg_acf_x[lag-1] - (1/np.e)) / \
                                  (avg_acf_x[lag-1] - avg_acf_x[lag])
            else: # If no interpolation possible, take current lag
                 theta_1_pixels = float(lag)
            break
            
    if np.isnan(theta_1_pixels): # Fallback if 1/e drop not found
        theta_1_pixels = max_lag_x / 2.0 
        print(f"Warning: theta_1 (x-corr length) estimation fallback: {theta_1_pixels:.2f} grid cells.")
    if theta_1_pixels < 1.0: theta_1_pixels = 1.0 # Ensure minimum length

    # --- Estimate theta_2 (correlation length in y-direction) ---
    max_lag_y = n_rows // 2
    if max_lag_y < 1: max_lag_y = 1

    all_acfs_y = []
    for j in range(n_cols): # Iterate over each column
        col_data = field_component[:, j]
        if np.sum(~np.isnan(col_data)) > 1:
            acf_y = _calculate_acf_1d(col_data, nlags=max_lag_y)
            all_acfs_y.append(acf_y)

    avg_acf_y = np.full(max_lag_y + 1, np.nan)
    if all_acfs_y:
        avg_acf_y = np.nanmean(np.array(all_acfs_y), axis=0)

    theta_2_pixels = np.nan
    for lag in range(1, len(avg_acf_y)):
        if not np.isnan(avg_acf_y[lag]) and avg_acf_y[lag] < (1/np.e):
            if not np.isnan(avg_acf_y[lag-1]) and avg_acf_y[lag-1] > (1/np.e) and \
               (avg_acf_y[lag-1] - avg_acf_y[lag] > 1e-9):
                theta_2_pixels = (lag - 1) + (avg_acf_y[lag-1] - (1/np.e)) / \
                                 (avg_acf_y[lag-1] - avg_acf_y[lag])
            else:
                theta_2_pixels = float(lag)
            break
            
    if np.isnan(theta_2_pixels): # Fallback
        theta_2_pixels = max_lag_y / 2.0
        print(f"Warning: theta_2 (y-corr length) estimation fallback: {theta_2_pixels:.2f} grid cells.")
    if theta_2_pixels < 1.0: theta_2_pixels = 1.0

    # Convert from grid cells to physical units using grid spacing
    theta_1 = theta_1_pixels * dx
    theta_2 = theta_2_pixels * dy
    
    # Final safety net if values are still NaN (e.g., if dx/dy were problematic or field was all NaN)
    if np.isnan(theta_1):
        theta_1 = (n_cols / 4.0) * (dx if dx > 0 else 1.0) # Use a fraction of domain size
        print(f"Critical Warning: theta_1 defaulted to {theta_1:.2f} (domain fraction).")
    if np.isnan(theta_2):
        theta_2 = (n_rows / 4.0) * (dy if dy > 0 else 1.0)
        print(f"Critical Warning: theta_2 defaulted to {theta_2:.2f} (domain fraction).")

    return theta_1, theta_2

from scipy.optimize import curve_fit

def compute_power_spectrum(field):
    """
    Compute the 2D power spectrum of a field using FFT, handling NaNs.
    Args:
        field (2D array): input scalar field (e.g., velocity component or magnitude).
    Returns:
        psd2D (2D array): power spectral density.
    """
    # Handle NaNs: subtract nanmean, then fill NaNs with 0 for FFT
    field_mean = np.nanmean(field)
    if np.isnan(field_mean): # Handles case where field is all NaNs
        field_mean = 0.0

    f_centered = field - field_mean
    # Replace NaNs with 0.0 for FFT processing
    f_filled = np.nan_to_num(f_centered, nan=0.0)

    # 2D FFT
    F = np.fft.fft2(f_filled)
    # Shift zero freq to center
    F_shift = np.fft.fftshift(F)
    # Power spectral density
    # Normalizing by the total number of points in the (potentially zero-filled) array
    psd2D = np.abs(F_shift)**2 / f_filled.size
    return psd2D

def radial_average(psd2D, dx=1.0):
    """
    Compute radial average of a 2D power spectral density.
    Args:
        psd2D (2D array): power spectral density.
        dx (float): spatial sampling interval.
    Returns:
        k (1D array): radial frequency bins.
        psd_rad (1D array): radially averaged PSD.
    """
    ny, nx = psd2D.shape
    # Fourier frequencies
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    # Bin edges
    k_bins = np.linspace(0, K.max(), min(nx, ny)//2)
    psd_rad = np.zeros(len(k_bins)-1)
    k_vals = 0.5*(k_bins[:-1] + k_bins[1:])
    # Digitize
    inds = np.digitize(K.ravel(), k_bins)
    for i in range(1, len(k_bins)):
        mask = inds == i
        if mask.sum()>0:
            psd_rad[i-1] = psd2D.ravel()[mask].mean()
    return k_vals, psd_rad

def theoretical_spectrum(k, theta, Ef, a0=1.0):
    """
    Theoretical PSD model: Ef + a0*sqrt(pi/2)*theta*exp(-pi^2*k^2*theta^2/2)
    Args:
        k (1D): radial frequencies.
        theta (float): correlation length.
        Ef (float): noise floor.
        a0 (float): variance amplitude (optional).
    """
    return Ef + a0 * np.sqrt(np.pi/2) * theta * np.exp(- (np.pi**2) * k**2 * theta**2 / 2)

def estimate_theta_fsv(field, dx=1.0, initial_guess=(1.0, 1e-3)):
    """
    Estimate correlation length theta using Frequency-domain Sample Variogram (FSV) for 2D field.
    Handles NaN values in the input field.
    Args:
        field (2D array): scalar field.
        dx (float): spatial sampling interval.
        initial_guess (tuple): initial guess for (theta, Ef).
    Returns:
        popt (tuple): fitted parameters (theta, Ef, a0). Returns (np.nan, np.nan, np.nan) if fit fails.
        pcov (2D): covariance of fit. Returns np.inf if fit fails.
    """
    if np.all(np.isnan(field)):
        print("Warning: Input field to estimate_theta_fsv is all NaN.")
        return (np.nan, np.nan, np.nan), np.full((3, 3), np.inf)

    # Compute PSD (handles NaNs by filling with 0 after demeaning)
    psd2D = compute_power_spectrum(field)
    
    # Radial average
    k_vals, psd_rad = radial_average(psd2D, dx=dx)

    # Filter out NaN/inf values from psd_rad and corresponding k_vals for curve_fit
    # Also ensure psd_rad is positive and finite, as theoretical_spectrum expects positive values.
    valid_indices = (
        ~np.isnan(psd_rad) & ~np.isinf(psd_rad) & (psd_rad > 1e-12) &  # psd_rad should be positive and non-zero
        ~np.isnan(k_vals) & ~np.isinf(k_vals)
    )

    if np.sum(valid_indices) < 3:  # Need at least 3 points for 3 parameters (theta, Ef, a0)
        print(f"Warning: Not enough valid data points ({np.sum(valid_indices)}) for curve_fit in estimate_theta_fsv after radial average.")
        return (np.nan, np.nan, np.nan), np.full((3, 3), np.inf)

    k_vals_fit = k_vals[valid_indices]
    psd_rad_fit = psd_rad[valid_indices]

    # Initial guess for a0 (variance) using nanvar
    initial_a0 = np.nanvar(field)
    if np.isnan(initial_a0) or initial_a0 <= 1e-12: # Handle case where variance is zero/NaN or too small
        initial_a0 = 1.0 # Default fallback for a0
        if np.nanvar(field) <= 1e-12 and not np.isnan(np.nanvar(field)):
            print("Warning: Low variance in field for estimate_theta_fsv, initial_a0 may be sensitive.")


    p0 = list(initial_guess) + [initial_a0]
    
    try:
        # Fit theoretical model to PSD
        popt, pcov = curve_fit(theoretical_spectrum, k_vals_fit, psd_rad_fit,
                               p0=p0,
                               bounds=([0, 0, 1e-9], [np.inf, np.inf, np.inf]), # a0 should be > 0
                               maxfev=5000) 
        theta, Ef, a0 = popt
        return (theta, Ef, a0), pcov
    except (RuntimeError, ValueError) as e:
        print(f"Warning: curve_fit failed in estimate_theta_fsv: {e}")
        return (np.nan, np.nan, np.nan), np.full((3, 3), np.inf)


# Smooth U, V with kriging
def krige_velocity_field(X, Y, U, V):
    """
    Apply kriging interpolation to velocity field components.
    
    Parameters:
    -----------
    X, Y : ndarray
        Coordinates of velocity vectors
    U, V : ndarray
        Velocity components to be kriged
        
    Returns:
    --------
    U_kriged, V_kriged : ndarray
        Kriged velocity fields
    """
    
    mu_u = np.nanmean(U)
    mu_v = np.nanmean(V)

    sigma_u = np.nanstd(U)
    sigma_v = np.nanstd(V)
    print(f'{mu_u=}, {mu_v=}, {sigma_u=}, {sigma_v=}')

    # Estimate correlation lengths theta_1 and theta_2 using the U component
    # Alternatively, you could use V, or average estimates from U and V.
    # Or estimate theta_1 from U's x-correlation and theta_2 from U's y-correlation.
    # The function estimate_correlation_lengths_from_field does this.
    print("Estimating correlation lengths theta_1 and theta_2...")
    # theta_1, theta_2 = estimate_correlation_lengths_from_field(U, X, Y)
    # theta_3, theta_4 = estimate_correlation_lengths_from_field(V, X, Y)
    # theta_1, theta_2, _ = estimate_theta_fsv(U, dx=16.0)[0]
    # theta_3, theta_4, _ = estimate_theta_fsv(V, dx=16.0)[0]
    theta_1, theta_2, theta_3, theta_4 = 16, 16, 16, 16
    print(f"Estimated theta_1: {theta_1:.2f}, theta_2: {theta_2:.2f} (in X/Y units)")
    print(f"Estimated theta_3: {theta_3:.2f}, theta_4: {theta_4:.2f} (in X/Y units)")

    # Convert to 1D arrays
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    U_flat = U.flatten()
    V_flat = V.flatten()

    # Create mask for valid data points
    mask = np.isfinite(U_flat) & np.isfinite(V_flat)
    X_flat = X_flat[mask]
    Y_flat = Y_flat[mask]
    U_flat = U_flat[mask]
    V_flat = V_flat[mask]

    # Calculate the distance matrix
    x_dist = np.abs(X_flat[:, None] - X_flat[None, :])
    print(x_dist[1, 0])
    y_dist = np.abs(Y_flat[:, None] - Y_flat[None, :])
    dist_matrix_u = np.sqrt((x_dist/theta_1)**2 + (y_dist/theta_2)**2)
    dist_matrix_v = np.sqrt((x_dist/theta_3)**2 + (y_dist/theta_4)**2)

    # Create the covariance matrix using the Gaussian model
    P_u = sigma_u**2 * (np.exp(-(dist_matrix_u / (2))**2))
    P_v = sigma_v**2 * (np.exp(-(dist_matrix_v / (2))**2))

    assert np.allclose(P_u, P_u.T), "Covariance matrix is not symmetric"
    assert np.allclose(sigma_u**2, np.diag(P_u)), "Diagonal elements of covariance matrix are not equal to variance"

    # Create the error covariance matrix
    eps_dt = 1e-9 # assumed error in time of 1 ns
    eps_dx = 0.5*l_px # assumed error in distance of 1 pixel
    eps_u = 1/dt*np.sqrt((U_flat*eps_dt)**2 + (eps_dx)**2)
    eps_v = 1/dt*np.sqrt((V_flat*eps_dt)**2 + (eps_dx)**2)
    R_u = np.diag(eps_u**2)#*1e-9
    R_v = np.diag(eps_v**2)#*1e-9

    u_kriged = U.copy()
    v_kriged = V.copy()

    mask_2d = ~np.isnan(U)

    u_kriged[mask_2d] = mu_u +  P_u @ np.linalg.inv(P_u + R_u)@(U_flat - mu_u)
    v_kriged[mask_2d] = mu_v + P_v @ np.linalg.inv(P_v + R_v)@(V_flat - mu_v)
    
    print(np.linalg.norm(u_kriged[~np.isnan(u_kriged)] - U[~np.isnan(u_kriged)]))
    print(np.linalg.norm(v_kriged[~np.isnan(v_kriged)] - V[~np.isnan(v_kriged)]))
    return u_kriged, v_kriged

# Apply kriging to smooth velocity fields
krig=False
if krig:
    print("Applying kriging to velocity fields...")
    U, V = krige_velocity_field(X, Y, U, V)
    # # compare with original

# # U_kriged[mask_1==1] = np.nan
# # V_kriged[mask_1==1] = np.nan

# print(np.linalg.norm(U_kriged[~np.isnan(U_kriged)] - U[~np.isnan(U_kriged)]))
# print(np.linalg.norm(V_kriged[~np.isnan(V_kriged)] - V[~np.isnan(V_kriged)]))

U = U[1:-1, 1:-1]
V = V[1:-1, 1:-1]
# U = nanfilter(U, sigma=sigma_from_filter(4))
# V = nanfilter(V, sigma=sigma_from_filter(4))
refU = reference_data[:, 2]
refV = reference_data[:, 3]
refValid = reference_data[:, 4]

refU[refValid == 0] = np.nan
refV[refValid == 0] = np.nan
rows, cols = 78, 101
refU = refU.reshape((rows, cols))
refV = refV.reshape((rows, cols))
refMag = np.sqrt(refU**2 + refV**2)
fig, axs = plt.subplots(2, 3, figsize=(15, 5))
U_mag = np.sqrt(U**2 + V**2)

# Top row with consistent color scale
im0 = axs[0,0].imshow(U_mag, cmap='jet', vmin=np.nanmin(refMag), vmax=np.nanmax(refMag))
axs[0,0].set_title('U Magnitude')
axs[0,0].set_aspect('equal')

im1 = axs[0,1].imshow(U, cmap='jet', vmin=np.nanmin(refU), vmax=np.nanmax(refU))
axs[0,1].set_title('U Velocity')
axs[0,1].set_aspect('equal')

im2 = axs[0,2].imshow(V, cmap='jet', vmin=np.nanmin(refV), vmax=np.nanmax(refV))
axs[0,2].set_title('V Velocity')
axs[0,2].set_aspect('equal')

# Bottom row with same color scale
im3 = axs[1,0].imshow(refMag, cmap='jet', vmin=np.nanmin(refMag), vmax=np.nanmax(refMag))
# axs[1,0].set_title('U Magnitude')
axs[1,0].set_aspect('equal')

im4 = axs[1,1].imshow(refU, cmap='jet', vmin=np.nanmin(refU), vmax=np.nanmax(refU))
# axs[1,1].set_title('U Velocity')
axs[1,1].set_aspect('equal')

im5 = axs[1,2].imshow(refV, cmap='jet', vmin=np.nanmin(refV), vmax=np.nanmax(refV))
# axs[1,2].set_title('V Velocity')
axs[1,2].set_aspect('equal')

# Add separate colorbars for each column
cbar0 = fig.colorbar(im0, ax=[axs[0,0], axs[1,0]])
cbar0.set_label('Velocity Magnitude (m/s)')

cbar1 = fig.colorbar(im1, ax=[axs[0,1], axs[1,1]])
cbar1.set_label('U Velocity (m/s)')

cbar2 = fig.colorbar(im2, ax=[axs[0,2], axs[1,2]])
cbar2.set_label('V Velocity (m/s)')

# plt.tight_layout()
if krig:
    plt.savefig('piv_comparison_kriging.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig('piv_comparison_no_kriging.png', dpi=300, bbox_inches='tight')
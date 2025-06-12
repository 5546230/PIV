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

# Store original images
pic_1_orig = data[:mid//2, :].copy()
pic_2_orig = data[mid//2:, :].copy()

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(pic_1, cmap='gray')
# axs[0].set_title('Image 1')
# axs[1].imshow(pic_2, cmap='gray')
# axs[1].set_title('Image 2')
# plt.show()


l_px = 170e-3/1616 # length of a pixel in meters #TODO: check this value
dt = 70e-6 # time between images in seconds

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
            x_peak -= window_size - 1
            y_peak -= window_size - 1
            
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
    U_orig = U.copy()
    V_orig = V.copy()
    eps = 0.1 * l_px / dt  # Small epsilon to avoid division by zero
    for i_idx in range(1, rows - 1):
        for j_idx in range(1, cols - 1):
            if mask_ds is not None and mask_ds[i_idx, j_idx]:
                continue
            # Get the 10x10 neighborhood
            u_neigh = U[i_idx-1:i_idx+2, j_idx-1:j_idx+2].flatten()
            v_neigh = V[i_idx-1:i_idx+2, j_idx-1:j_idx+2].flatten()

            # extract the 3x3 neighborhood excluding the center
            mask = np.ones((3,3), dtype=bool)
            mask[1,1] = False  # Center element
            u_neigh_2 = U[i_idx-1:i_idx+2, j_idx-1:j_idx+2][mask].flatten()
            v_neigh_2 = V[i_idx-1:i_idx+2, j_idx-1:j_idx+2][mask].flatten()

            # Calculate the median and standard deviation
            u_median = np.nanmedian(u_neigh_2)
            v_median = np.nanmedian(v_neigh_2)

            # Calculate the fluctuation relative to the median
            u_fluct = U[i_idx, j_idx] - u_median
            v_fluct = V[i_idx, j_idx] - v_median

            # Calculate the residual
            u_res = u_neigh_2 - u_median
            v_res = v_neigh_2 - v_median

            # Calculate the median of the absolute residuals
            u_res_median = np.nanmedian(np.abs(u_res))
            v_res_median = np.nanmedian(np.abs(v_res))

            # Calculate the normalized fluctuations
            u_norm_fluct = np.abs(u_fluct / (u_res_median + eps))
            v_norm_fluct = np.abs(v_fluct / (v_res_median + eps))
            norm_fluct = np.sqrt(u_norm_fluct**2 + v_norm_fluct**2)

            # Check if the current value is an outlier
            if norm_fluct > 3.5:
                U[i_idx, j_idx] = np.nan
                V[i_idx, j_idx] = np.nan
                # snrs[i_idx, j_idx] = np.nan


    return U, V, X, Y, snrs, U_orig, V_orig


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
    # for _ in range(3): # Iterative outlier removal
    #     U_temp = U_pix.copy()
    #     V_temp = V_pix.copy()
    #     for i_r in range(rows):
    #         for j_c in range(cols):
    #             if mask_ds is not None and i_r < mask_ds.shape[0] and j_c < mask_ds.shape[1] and mask_ds[i_r, j_c]:
    #                 continue
    #             if np.isnan(U_pix[i_r, j_c]):
    #                 continue

    #             # Define neighborhood (e.g., 3x3 or 5x5, excluding NaNs and masked points)
    #             u_neigh, v_neigh = [], []
    #             # Using a 3x3 neighborhood for simplicity here. For larger, adjust bounds.
    #             for ni in range(max(0, i_r-1), min(rows, i_r+2)):
    #                 for nj in range(max(0, j_c-1), min(cols, j_c+2)):
    #                     if (ni == i_r and nj == j_c):
    #                         continue
    #                     if mask_ds is not None and ni < mask_ds.shape[0] and nj < mask_ds.shape[1] and mask_ds[ni, nj]:
    #                         continue
    #                     if not np.isnan(U_temp[ni, nj]): # Use U_temp for consistent comparison within iteration
    #                         u_neigh.append(U_temp[ni, nj])
    #                         v_neigh.append(V_temp[ni, nj])
                
    #             if not u_neigh: # No valid neighbors
    #                 continue

    #             u_median = np.median(u_neigh)
    #             v_median = np.median(v_neigh)
                
    #             # Normalized median test (from Westerweel & Scarano 2005)
    #             # Fluctuations of u relative to median of neighbors
    #             u_fluct = [u - u_median for u in u_neigh]
    #             v_fluct = [v - v_median for v in v_neigh]
                
    #             median_u_fluct = np.median(np.abs(u_fluct))
    #             median_v_fluct = np.median(np.abs(v_fluct))

    #             # Threshold (e.g., 2 times the median fluctuation)
    #             # Using a simpler std-dev based approach from original code for now
    #             u_std = np.std(u_neigh) if len(u_neigh) > 1 else 0
    #             v_std = np.std(v_neigh) if len(v_neigh) > 1 else 0

    #             # If std is very small, avoid overly aggressive filtering
    #             u_std = max(u_std, 1e-2) # Minimum std deviation to consider
    #             v_std = max(v_std, 1e-2)


    #             if np.abs(U_pix[i_r, j_c] - u_median) > 2 * u_std or \
    #                np.abs(V_pix[i_r, j_c] - v_median) > 2 * v_std:
    #                 U_pix[i_r, j_c] = u_median # Replace with median
    #                 V_pix[i_r, j_c] = v_median
    #                 # Or, could set to NaN if preferred:
    #                 # U_pix[i_r, j_c] = np.nan
    #                 # V_pix[i_r, j_c] = np.nan


    # Convert pixel displacements to physical velocities
    U = -U_pix * l_px / dt
    V = V_pix * l_px / dt # Negative sign for V if y-axis of image is downward and physical V is upward

    return U, V, X, Y, snrs_map

snr_threshold = 1.
reference_data = np.loadtxt('data\\alpha_15_100_PIV_MP(3x32x32_50ov)=unknown\\B00006.dat', skiprows=3)
refX = reference_data[:, 0]
refY = reference_data[:, 1]
# calculate pixel length from processed data.
l_px = (refX.max() - refX.min()) / (pic_1.shape[1] ) * 1e-3
print(f'{l_px=}')

dt = 70e-6
window_size = 32
# Example usage (replace the manual PIV computation with function call):
mask, mask_ds = apply_velocity_mask_to_piv(window_size, 50)
U, V, X, Y, snrs, U_orig, V_orig = calculate_piv(pic_1, pic_2, window_size, l_px, dt, mask_ds, snr_threshold, 50)
# U, V, X, Y, snrs = calculate_piv_multipass(pic_1, pic_2, window_size, l_px, dt,3, mask_ds, snr_threshold, 50)
U[mask_ds==1], V[mask_ds==1] = np.nan, np.nan
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_results(X, Y, U, V, file_name='piv_results.png', step=3):
    """
    Plot the PIV results with quiver and contour plots.
    
    Parameters:
    -----------
    X, Y : ndarray
        Coordinates of velocity vectors
    U, V : ndarray
        Horizontal and vertical velocity components
    file_name : str
        Name of the file to save the plot
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.contourf(X, Y, np.sqrt(U**2 + V**2), cmap='jet', corner_mask=False, levels=12)
    
    # Create colorbar with same height as axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = fig.colorbar(im, cax=cax)
    
    Xds = X[::step, ::step]
    Yds = Y[::step, ::step]
    Uds = U[::step, ::step]
    Vds = V[::step, ::step]
    ax.quiver(Xds, Yds, Uds, Vds)
    
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    cbar.set_label('Velocity Magnitude (m/s)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(file_name[:-3] + 'pdf' , format='pdf')
    plt.close(fig)
    # plt.show()


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
# fig, axs = plt.subplots(2, 3, figsize=(15, 5))
U_mag = np.sqrt(U**2 + V**2)
X, Y = X[1:-1, 1:-1], Y[1:-1, 1:-1]
X = X * l_px * 1e3  # Convert to mm
Y = Y * l_px * 1e3  # Convert to mm
refX = refX.reshape((rows, cols)) - np.min(refX)
refY = refY.reshape((rows, cols)) - np.min(refY)
plot_results(X, Y, U, -V, file_name='figures/piv_results_own_code.png', step=3)
plot_results(refX, refY[::-1, :], refU, -refV, file_name='figures/piv_results_reference.png', step=3)

# plot data for all angles of attack 
angles = [0, 5, 15]

for angle in angles:
    # Load the reference data for the current angle
    if angle == 0:
        reference_data = np.loadtxt(f'data/alpha_0_20_SubOverTimeMin_sL=all_01_PIV_MP(3x32x32_50ov)=unknown/B00001.dat', skiprows=3)
    elif angle == 5:
        reference_data = np.loadtxt(f'data/alpha_5_20_PIV_MP(3x32x32_50ov)=unknown/B00001.dat', skiprows=3)
    else:
        reference_data = np.loadtxt(f'data/alpha_{angle}_100_PIV_MP(3x32x32_50ov)=unknown/B00001.dat', skiprows=3)

    refX = reference_data[:, 0]
    refY = reference_data[:, 1]
    refU = reference_data[:, 2]
    refV = reference_data[:, 3]
    refValid = reference_data[:, 4]

    refX = refX - np.min(refX)  # Normalize X
    refY = refY - np.min(refY)  # Normalize Y
    rows, cols = 78, 101  # Assuming these are the dimensions of the reference data

    # Reshape and filter the reference data
    refU[refValid == 0] = np.nan
    refV[refValid == 0] = np.nan
    refU = refU.reshape((rows, cols))
    refV = refV.reshape((rows, cols))
    
    # Calculate the magnitude of the reference velocity
    refMag = np.sqrt(refU**2 + refV**2)
    
    # Plot the results for the current angle
    plot_results(refX.reshape((rows, cols)), 
                 refY.reshape((rows, cols))[::-1, :], 
                 refU, -refV, 
                 file_name=f'figures/piv_results_{angle}_instantaneous.png', step=3)
    
for angle in angles:
    # Load the reference data for the current angle
    if angle == 0:
        reference_data = np.loadtxt(f'data/alpha_0_20_SubOverTimeMin_sL=all_01_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00001.dat', skiprows=3)
    elif angle == 5:
        reference_data = np.loadtxt(f'data/alpha_5_20_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00001.dat', skiprows=3)
    else:
        reference_data = np.loadtxt(f'data/alpha_{angle}_100_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00001.dat', skiprows=3)

    refX = reference_data[:, 0]
    refY = reference_data[:, 1]
    refU = reference_data[:, 2]
    refV = reference_data[:, 3]
    refValid = reference_data[:, 4]

    refX = refX - np.min(refX)  # Normalize X
    refY = refY - np.min(refY)  # Normalize Y
    rows, cols = 78, 101  # Assuming these are the dimensions of the reference data

    # Reshape and filter the reference data
    refU[refValid == 0] = np.nan
    refV[refValid == 0] = np.nan
    refU = refU.reshape((rows, cols))
    refV = refV.reshape((rows, cols))
    refX = refX.reshape((rows, cols))
    refY = refY.reshape((rows, cols))
    refY = refY[::-1, :]  # Reverse Y-axis for correct orientation
    
    # Calculate the magnitude of the reference velocity
    refMag = np.sqrt(refU**2 + refV**2)
    
    # Plot the results for the current angle
    plot_results(refX, 
                 refY, 
                 refU, -refV, 
                 file_name=f'figures/piv_results_{angle}_mean.png', step=3)
    
for angle in angles:
    # Load the reference data for the current angle
    if angle == 0:
        reference_data_mean = np.loadtxt(f'data/alpha_0_20_SubOverTimeMin_sL=all_01_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00001.dat', skiprows=3)
        reference_data_stdev = np.loadtxt(f'data/alpha_0_20_SubOverTimeMin_sL=all_01_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00002.dat', skiprows=3)
    elif angle == 5:
        reference_data_mean = np.loadtxt(f'data/alpha_5_20_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00001.dat', skiprows=3)
        reference_data_stdev = np.loadtxt(f'data/alpha_5_20_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00002.dat', skiprows=3)
    else:
        reference_data_mean = np.loadtxt(f'data/alpha_{angle}_100_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00001.dat', skiprows=3)
        reference_data_stdev = np.loadtxt(f'data/alpha_{angle}_100_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00002.dat', skiprows=3)

    refX = reference_data_mean[:, 0]
    refY = reference_data_mean[:, 1][np.isclose(refX, 120, atol=0.5)][1:-1]
    refUmean = reference_data_mean[:, 2][np.isclose(refX, 120, atol=0.5)][1:-1]
    # plt.plot(refUmean, refY-50, label=f'alpha={angle} deg')
    refUstdev = reference_data_stdev[:, 2][np.isclose(refX, 120, atol=0.5)][1:-1]
    # plt.plot(refUstdev, refY-50, linestyle='--', label=f'alpha={angle} deg (stdev)')
    data_array = np.column_stack((refY-50, refUmean, refUstdev))
    np.savetxt(f'data/alpha_{angle}_Uprov.txt', data_array, delimiter='\t', header='y_mm\tmean_u\trms_u', comments='')
# plt.legend()
# plt.show()

    # Load the reference data for the current angle
for N in [16, 32, 64]:
    reference_data = np.loadtxt(f'data/alpha_0_20_SubOverTimeMin_sL=all_01_PIV_SP({N}x{N}_0ov)=unknown/B00001.dat', skiprows=3)
    refX = reference_data[:, 0]
    refY = reference_data[:, 1]
    refU = reference_data[:, 2]
    refV = reference_data[:, 3]
    refValid = reference_data[:, 4]

    refX = refX - np.min(refX)  # Normalize X
    refY = refY - np.min(refY)  # Normalize Y
    if N == 16:
        rows, cols = 78, 101
    elif N == 32: 
        rows, cols = 39, 51 
    else:  # N == 64
        rows, cols = 20, 26

    # Reshape and filter the reference data
    refU[refValid == 0] = np.nan
    refV[refValid == 0] = np.nan
    refU = refU.reshape((rows, cols))
    refV = refV.reshape((rows, cols))

    # Calculate the magnitude of the reference velocity
    refMag = np.sqrt(refU**2 + refV**2)

    # Plot the results for the current angle
    plot_results(refX.reshape((rows, cols)), 
                    refY.reshape((rows, cols))[::-1, :], 
                    refU, -refV, 
                    file_name=f'figures/piv_results_sp_{N}.png', step=3)
 
# check overlap effect on PIV results
reference_data = np.loadtxt('data/alpha_0_20_SubOverTimeMin_sL=all_01_PIV_SP(32x32_50ov)=unknown/B00001.dat', skiprows=3)
refX = reference_data[:, 0]
refY = reference_data[:, 1]
refU = reference_data[:, 2]
refV = reference_data[:, 3]
refValid = reference_data[:, 4]
refX = refX - np.min(refX)  # Normalize X
refY = refY - np.min(refY)  # Normalize Y
rows, cols = 78, 101  # Assuming these are the dimensions of the reference data
# Reshape and filter the reference data
refU[refValid == 0] = np.nan
refV[refValid == 0] = np.nan
refU = refU.reshape((rows, cols))
refV = refV.reshape((rows, cols))
# Plot the results for the current angle
plot_results(refX.reshape((rows, cols)), 
             refY.reshape((rows, cols))[::-1, :], 
             refU, -refV, 
             file_name=f'figures/piv_results_sp_32_50ov.png', step=3)

# check short dt at alfa 15
reference_data = np.loadtxt('data/alpha_15_20_dt_6_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00001.dat', skiprows=3)
refX = reference_data[:, 0]
refY = reference_data[:, 1]
refU = reference_data[:, 2]
refV = reference_data[:, 3]

refValid = reference_data[:, 4]
refX = refX - np.min(refX)  # Normalize X
refY = refY - np.min(refY)  # Normalize Y
rows, cols = 78, 101  # Assuming these are the dimensions of the reference data
# Reshape and filter the reference data
refU[refValid == 0] = np.nan
refV[refValid == 0] = np.nan
refU = refU.reshape((rows, cols))
refV = refV.reshape((rows, cols))
# Plot the results for the current angle
plot_results(refX.reshape((rows, cols)), 
             refY.reshape((rows, cols))[::-1, :], 
             refU, -refV, 
             file_name=f'figures/piv_results_short_dt.png', step=3)

# calculate the mean of the PIV results for angle 15
# N_ensemble = 20
for N_ensemble in (5, 10, 20, 50):
    reference_U = np.zeros((78, 101))
    reference_V = np.zeros((78, 101))
    denominator = np.zeros((78, 101))
    for i in range(N_ensemble):
        n_str = str(i+1).zfill(2)
        reference_data = np.loadtxt(f'data/alpha_15_100_PIV_MP(3x32x32_50ov)=unknown/B000{n_str}.dat', skiprows=3)
        refX = reference_data[:, 0]
        refY = reference_data[:, 1]
        refU = reference_data[:, 2]
        refV = reference_data[:, 3]

        reference_U += refU.reshape((78, 101))
        reference_V += refV.reshape((78, 101))
        refValid = reference_data[:, 4]
        denominator += (refValid.reshape((78, 101)) > 0).astype(float)

    reference_U /= denominator
    reference_V /= denominator

    # Plot the mean results for angle 15
    plot_results(refX.reshape((rows, cols)), 
                refY.reshape((rows, cols))[::-1, :], 
                reference_U, -reference_V, 
                file_name=f'figures/piv_results_mean_angle_15_{N_ensemble}.png', step=3)
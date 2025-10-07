import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pylab as plt

def hogbom_clean(dirty_image, psf, gain=0.1, threshold=0.001, max_iterations=1000):
    """
    Performs the Hogbom CLEAN algorithm on a dirty image.

    Parameters:
    dirty_image (2D numpy array): The dirty image to be cleaned.
    psf (2D numpy array): The point spread function (PSF) or dirty beam.
    gain (float): The loop gain (fraction of the peak to subtract each iteration). Default is 0.1.
    threshold (float): The stopping threshold for the peak residual. Default is 0.001.
    max_iterations (int): Maximum number of iterations to perform. Default is 1000.

    Returns:
    clean_image (2D numpy array): The cleaned image.
    residual (2D numpy array): The final residual image.
    """
    # Initialize residual and clean component images
    residual = dirty_image.copy()
    clean_components = np.zeros_like(dirty_image)

    # Find the center of the PSF
    psf_center = (psf.shape[0] // 2, psf.shape[1] // 2)

    for iteration in range(max_iterations):
        print(f'iteration: {iteration}')
        # Find the peak (maximum absolute value) in the residual image
        max_val = residual.max()
        min_val = residual.min()

        if abs(max_val) >= abs(min_val):
            peak = max_val
            y_peak, x_peak = np.unravel_index(residual.argmax(), residual.shape)
        else:
            peak = min_val
            y_peak, x_peak = np.unravel_index(residual.argmin(), residual.shape)

        # Check if the peak is below the threshold
        if abs(peak) < threshold:
            print(f"Stopping at iteration {iteration} with peak residual {peak}")
            break

        # Update the clean components image
        clean_components[y_peak, x_peak] += gain * peak

        # Scale the PSF by the gain and peak value
        scaled_psf = gain * peak * psf

        # Calculate the coordinates for the overlapping region
        y_min = max(0, y_peak - psf_center[0])
        y_max = min(residual.shape[0], y_peak + (psf.shape[0] - psf_center[0]))
        x_min = max(0, x_peak - psf_center[1])
        x_max = min(residual.shape[1], x_peak + (psf.shape[1] - psf_center[1]))

        psf_y_min = max(0, psf_center[0] - y_peak)
        psf_y_max = psf_y_min + (y_max - y_min)
        psf_x_min = max(0, psf_center[1] - x_peak)
        psf_x_max = psf_x_min + (x_max - x_min)

        # Subtract the scaled PSF from the residual image
        residual[y_min:y_max, x_min:x_max] -= scaled_psf[psf_y_min:psf_y_max, psf_x_min:psf_x_max]

    # Create a clean beam (Gaussian)
    #fwhm = psf.shape[0] // 10  # You can adjust the FWHM as needed
    fwhm = 7  # eyeball fit to DSA-2000 W-config psf image
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    print(f'Gaussian restoring size of {psf.shape} and width {sigma}')
    clean_beam = gaussian_2d(psf.shape, sigma)

    # Convolve the clean components with the clean beam
    clean_image = gaussian_filter(clean_components, sigma=sigma)

    # Add the final residual to get the clean image
#    clean_image += residual

    return clean_image, residual

def gaussian_2d(shape, sigma):
    """
    Generates a 2D Gaussian kernel.

    Parameters:
    shape (tuple): Shape of the output array (height, width).
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    gaussian (2D numpy array): The Gaussian kernel.
    """
    y, x = np.indices(shape)
    x0 = shape[1] // 2
    y0 = shape[0] // 2
    gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    return gaussian

plot = False

if plot is True:
    fig = plt.figure()
    plt.imshow(np.abs(clean_image)**0.5, cmap='Greys')
    
    fig = plt.figure()
    plt.imshow(dirty_image, vmax=dirty_image.max()*1e-3, vmin=0)
    plt.show()
    
# Now, clean_image contains the deconvolved image

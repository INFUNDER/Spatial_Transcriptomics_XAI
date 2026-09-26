import numpy as np

def macenko_stain_normalization(img, Io=240, alpha=1, beta=0.15):
    """
    Normalize staining appearance of H&E stained images based on the Macenko algorithm.
    This implementation handles the OD space conversion and SVD to estimate stain vectors.
    """
    img = np.array(img).astype(np.float64)
    # Reshape image to (N, 3)
    img_shape = img.shape
    img_flat = img.reshape((-1, 3))
    
    # Calculate optical density (OD)
    # Add a small constant to avoid log(0)
    OD = -np.log10((img_flat + 1) / Io)
    
    # Filter background pixels
    # Calculate OD threshold based on beta
    OD_hat = OD[~np.any(OD < beta, axis=1)]
    
    # If all pixels are filtered out, return original image
    if OD_hat.shape[0] == 0:
        return np.clip(img, 0, 255).astype(np.uint8)
        
    # Estimate stain matrix using SVD
    # Calculate covariance matrix
    cov = np.cov(OD_hat, rowvar=False)
    
    # Eigen decomposition
    w, v = np.linalg.eigh(cov)
    
    # Project data onto the 2 largest eigenvectors
    top_2_eigenvectors = v[:, [1, 2]]
    proj = np.dot(OD_hat, top_2_eigenvectors)
    
    # Calculate angle of each projected point
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    
    # Find extreme angles
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)
    
    # Calculate stain vectors
    v_min = np.dot(top_2_eigenvectors, np.array([np.cos(min_phi), np.sin(min_phi)]))
    v_max = np.dot(top_2_eigenvectors, np.array([np.cos(max_phi), np.sin(max_phi)]))
    
    # Ensure Hematoxylin is first (has higher optical density in red channel)
    if v_min[0] > v_max[0]:
        HE = np.array([v_min, v_max])
    else:
        HE = np.array([v_max, v_min])
        
    # Standard reference stain matrix (from Macenko's original paper)
    HE_ref = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])
                       
    # Standard maximum stain concentrations
    max_C_ref = np.array([1.9705, 1.0308])
    
    # Calculate stain concentrations for current image
    C = np.linalg.lstsq(HE.T, OD_hat.T, rcond=None)[0]
    max_C = np.percentile(C, 99, axis=1)
    
    # Normalize concentrations
    C_norm = C * (max_C_ref / np.maximum(max_C, 1e-6))[:, None]
    
    # Reconstruct normalized image
    OD_norm = np.dot(HE_ref, C_norm)
    img_norm = Io * np.exp(-OD_norm * np.log(10))
    
    # Replace background pixels with original values
    img_norm_full = img_flat.copy()
    mask = ~np.any(OD < beta, axis=1)
    img_norm_full[mask] = img_norm.T
    
    img_norm_full = img_norm_full.reshape(img_shape)
    img_norm_full = np.clip(img_norm_full, 0, 255).astype(np.uint8)
    
    return img_norm_full

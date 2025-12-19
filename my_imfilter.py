import numpy as np

def my_imfilter(image, filter):
    """
    This function performs image filtering
    without using any built-in convolution functions.
    
    Input:
        image  - an H x W x C numpy array (float32 or float64)
        filter - a 2D numpy array (odd-sized kernel)
        
    Output:
        output - a filtered image with the same size as input
    """
    ############### YOUR CODE HERE ###############
    # Processing color images using a three-dimensional array

    h, w = filter.shape
    H, W, C = image.shape

    # Determine padding sizes
    pad_h = h // 2
    pad_w = w // 2

    # Zero padding
    image_pad = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    output = np.zeros_like(image)

    # Performing convolution
    for i in range (H):
        for j in range (W):
            for k in range (C):
                output[i,j,k] = np.sum( filter * image_pad[i : i+h, j : j+w ,k] )
    ############### YOUR CODE END ################
    return output

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

from encode import encode
from decode import decode
from encode_dct import encode_dct
from decode_dct import decode_dct, calculate_ber, calculate_metrics
from my_imfilter import my_imfilter

if __name__ == "__main__":
    # IMPORTANT: Use a lossless format
    cover_image = "../data/hotel_cal.bmp" 
    stego_image = "../result/stego.bmp" 
    filtered_stego_image = "../result/filtered_stego_image.bmp" 

    # --- ENCODING ---
    codeword = input("Enter the codeword: ")
    print("\nEmbedded Message:")
    print(codeword)
    
    try:
        encode(cover_image, codeword, stego_image)
    except FileNotFoundError:
        print(f"\n Error: Input image '{cover_image}' not found. Please provide an image file.")
        sys.exit(1)
    except Exception as e:
        print(f"\n An error occurred during encoding: {e}")
        sys.exit(1)

    # --- IMAGE FILTERING ---
    # 1. LOAD IMAGE: Must load the stego file into a NumPy array for filtering
    stego_image_np = cv2.imread(stego_image).astype(np.float64)
    if stego_image_np is None:
        raise FileNotFoundError(f"Could not load image file: {stego_image}")

    # 2. CREATE GAUSSIAN FILTER
    cutoff_frequency = 1
    # Kernel size must be an odd number (e.g., 25)
    ksize = int(cutoff_frequency * 4 + 1)
    
    # Create a 1D Gaussian kernel
    filter_1d = cv2.getGaussianKernel(ksize, cutoff_frequency)
    # Convert to a 2D separable Gaussian filter
    filter_2d = filter_1d @ filter_1d.T

    # 3. APPLY FILTER
    filtered_image_np = my_imfilter(stego_image_np, filter_2d)
    
    # 4. SAVE FILTERED IMAGE: Must save the NumPy array back to a file
    # Convert back to uint8 (0-255) for saving
    filtered_image_uint8 = np.clip(filtered_image_np, 0, 255).astype(np.uint8)
    cv2.imwrite(filtered_stego_image, filtered_image_uint8)
    
    # --- DECODING ---
    print("\n--- Decoding Result ---")
    result_1 = decode(stego_image)
    print("\nExtracted Message:")
    print(result_1)
    BER_1 = calculate_ber(codeword, result_1)
    


    result_2 = decode(filtered_stego_image)
    print("\nExtracted Message After Filtering:")
    print(result_2)
    BER_2 = calculate_ber(codeword, result_2)

    # --- DISPLAYING IMAGES ---
    cover_image = cv2.imread(cover_image)
    cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB) / 255.0
    cover_image = cv2.resize(cover_image, (0, 0), fx=0.5, fy=0.5)
    stego_image = cv2.imread(stego_image)
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB) / 255.0
    stego_image = cv2.resize(stego_image, (0, 0), fx=0.5, fy=0.5)
    filtered_stego_image = cv2.imread(filtered_stego_image)
    filtered_stego_image = cv2.cvtColor(filtered_stego_image, cv2.COLOR_BGR2RGB) / 255.0
    filtered_stego_image = cv2.resize(filtered_stego_image, (0, 0), fx=0.5, fy=0.5)
    plt.figure(1); plt.imshow(cover_image); plt.title("Cover image")
    plt.figure(2); plt.imshow(stego_image); plt.title("Encoded image")
    plt.figure(3); plt.imshow(filtered_stego_image); plt.title("Gaussian filter")

    psnr_value_1 = calculate_metrics(cover_image,stego_image)
    psnr_value_2 = calculate_metrics(cover_image,filtered_stego_image)
    print(f"PSNR: {psnr_value_1:.2f} dB")
    print(f"PSNR: {psnr_value_2:.2f} dB")

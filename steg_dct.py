import numpy as np
from scipy.fftpack import dct, idct
import cv2
import matplotlib.pyplot as plt

from encode_dct import encode_dct
from decode_dct import decode_dct, calculate_ber, calculate_metrics
from my_imfilter import my_imfilter


if __name__ == '__main__':

    cover_image = cv2.imread("../data/hotel_cal.jpg")
    cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB) / 255.0

    stego_image = "../result/stego_dct.jpg"
    filtered_stego_image = "../result/filtered_stego_image_dct.jpg"

    codeword = input("Enter the codeword: ")
    print("\nEmbedded Message:")
    print(codeword)

    encode_dct(cover_image, codeword, stego_image, beta=2)
    stego_image_np = cv2.imread(stego_image).astype(np.float64)
    
    

    # CREATE GAUSSIAN FILTER
    cutoff_frequency = 1
    # Kernel size must be an odd number (e.g., 25)
    ksize = int(cutoff_frequency * 4 + 1)
    
    # Create a 1D Gaussian kernel
    filter_1d = cv2.getGaussianKernel(ksize, cutoff_frequency)
    # Convert to a 2D separable Gaussian filter
    filter_2d = filter_1d @ filter_1d.T

    # APPLY FILTER
    filtered_image_np = my_imfilter(stego_image_np, filter_2d)
    
    # SAVE FILTERED IMAGE
    filtered_image_uint8 = np.clip(filtered_image_np, 0, 255).astype(np.uint8)
    cv2.imwrite(filtered_stego_image, filtered_image_uint8)

    print("\n--- Decoding Result ---")
    result_1 = decode_dct(stego_image)
    print("\nExtracted Message:")
    print(result_1)
    BER_1 = calculate_ber(codeword, result_1)
    print(f"Bit Error Rate (BER): {BER_1:.6f}")

    result_2 = decode_dct(filtered_stego_image)
    print("\nExtracted Message After Filtering:")
    print(result_2)
    BER_2 = calculate_ber(codeword, result_2)
    print(f"Bit Error Rate (BER): {BER_2:.6f}")



    # Display
    stego_image = cv2.imread(stego_image)
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB) / 255.0
    filtered_stego_image = cv2.imread(filtered_stego_image)
    filtered_stego_image = cv2.cvtColor(filtered_stego_image, cv2.COLOR_BGR2RGB) / 255.0
    plt.figure(1); plt.imshow(cover_image); plt.title('Cover image')
    plt.figure(2); plt.imshow(stego_image); plt.title('Encoded image')
    plt.figure(3); plt.imshow(filtered_stego_image); plt.title('Gaussian filter')

    psnr_value_1 = calculate_metrics(cover_image,stego_image)
    psnr_value_2 = calculate_metrics(cover_image,filtered_stego_image)
    print(f"PSNR: {psnr_value_1:.2f} dB")
    print(f"PSNR: {psnr_value_2:.2f} dB")




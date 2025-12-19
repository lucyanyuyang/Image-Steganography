import matplotlib.pyplot as plt
import numpy as np
import cv2

from encode_dct import encode_dct
from decode_dct import decode_dct, calculate_ber, calculate_metrics
from my_imfilter import my_imfilter

# Define sweep parameters
beta_values = np.arange(1, 8)
ber_filtered = []
psnr_filtered = []

cover_image = cv2.imread("../data/hotel_cal.jpg")
cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB) / 255.0

filtered_stego_image = "../result/filtered_stego_image_dct.jpg"

codeword = input("Enter the codeword: ")
print("\nEmbedded Message:")
print(codeword)

# Assuming cover_image and codeword are already defined in your script
print("--- Starting Filtered Image Analysis ---")

for beta in beta_values:
    # 1. Encode with current beta
    stego_path = f"../result/stego_beta_{beta}.jpg"
    encode_dct(cover_image, codeword, stego_path, beta=beta)
    
    # 2. Load Stego for filtering
    stego_np = cv2.imread(stego_path).astype(np.float64)
    

    # CREATE GAUSSIAN FILTER
    cutoff_frequency = 1
    # Kernel size must be an odd number (e.g., 25)
    ksize = int(cutoff_frequency * 4 + 1)
    
    # Create a 1D Gaussian kernel
    filter_1d = cv2.getGaussianKernel(ksize, cutoff_frequency)
    # Convert to a 2D separable Gaussian filter
    filter_2d = filter_1d @ filter_1d.T
    # 3. Apply your custom filter function
    # (Using the filter_2d Gaussian kernel defined previously)
    filtered_np = my_imfilter(stego_np, filter_2d)
    
    # 4. Save filtered result for decoding
    filtered_path = f"../result/filtered_beta_{beta}.jpg"
    filtered_uint8 = np.clip(filtered_np, 0, 255).astype(np.uint8)
    cv2.imwrite(filtered_path, filtered_uint8)
    
    # 5. Calculate Metrics for the Filtered Image
    # BER: How well the message survived the filter
    result = decode_dct(filtered_path)
    ber = calculate_ber(codeword, result)
    ber_filtered.append(ber)
    
    # PSNR: Quality of the filtered image compared to original cover
    filtered_rgb = cv2.cvtColor(filtered_uint8, cv2.COLOR_BGR2RGB) / 255.0
    p_val, _ = calculate_metrics(cover_image, filtered_rgb)
    psnr_filtered.append(p_val)
    
    print(f"Beta: {beta} | Filtered PSNR: {p_val:.2f} | Filtered BER: {ber:.4f}")

# --- Plotting Results ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left Axis: BER (Robustness)
ax1.set_xlabel('Beta (Embedding Strength)', fontsize=16)
ax1.set_ylabel('Bit Error Rate (BER)', color='black', fontsize=16)
ax1.plot(beta_values, ber_filtered, 'x-', color='black', label='BER (After Filter)')
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)

# Right Axis: PSNR (Quality)
ax2 = ax1.twinx()
ax2.set_ylabel('PSNR (dB)', color='black', fontsize=16)
ax2.plot(beta_values, psnr_filtered, 'o--', color='black', label='PSNR (After Filter)')
ax2.tick_params(axis='y', labelcolor='black', labelsize=16)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=20)

#plt.title('Impact of Beta on Filtered Image Robustness and Quality')
fig.tight_layout()
plt.show()

import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

# Standard JPEG luminance quantization table
Q = np.array([
 [16,11,10,16,24,40,51,61],
 [12,12,14,19,26,58,60,55],
 [14,13,16,24,40,57,69,56],
 [14,17,22,29,51,87,80,62],
 [18,22,37,56,68,109,103,77],
 [24,35,55,64,81,104,113,92],
 [49,64,78,87,103,121,120,101],
 [72,92,95,98,112,100,103,99]
], dtype=np.float32)


def dct2(a):  return dct(dct(a.T, norm='ortho').T, norm='ortho')

def decode_dct(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    rows, cols = arr.shape

    cx, cy = 3, 3
    TH = 0
    cnt = 0
    alpha = 0.95

    for i in range(0, rows-7, 8):
        for j in range(0, cols-7, 8):
            block = arr[i:i+8, j:j+8] - 128
            D = Q[cx,cy] * dct2(block)
            QD = np.round(D / Q)
            #print(QD[cx,cy])
            TH += (QD[cx,cy]/8)
            cnt += 1
            if cnt == 8:
                break
        if cnt == 8:
            break
    
    bits = ""
    skip = 8
    # 1) Read length
    for i in range(0, rows-7, 8):
        for j in range(0, cols-7, 8):

            if skip > 0:
                skip -= 1
                continue

            block = arr[i:i+8, j:j+8] - 128
            D = Q[cx,cy] * dct2(block)
            QD = np.round(D / Q)
            #print(QD[cx,cy]) 
            if (QD[cx, cy] > TH):
                bits += str(1)
                TH = alpha * TH + (1-alpha) * QD[cx, cy]
            else:
                bits += str(0) 
                TH = alpha * TH + (1-alpha) * QD[cx, cy] 
            
            if len(bits) == 16:
                break
        if len(bits) == 16:
            break

    msg_len = int(bits, 2)
    print("\nThe message length from the header:")
    print(msg_len)
    msg_bits = msg_len * 8

    # 2) extract message bits
    bits = ""
    skip = 24
    collected = 0

    for i in range(0, rows-7, 8):
        for j in range(0, cols-7, 8):

            if skip > 0:
                skip -= 1
                continue

            if collected >= msg_bits:
                break

            block = arr[i:i+8, j:j+8] - 128
            D = Q[cx,cy] * dct2(block)
            QD = np.round(D / Q)
            #print(QD[cx,cy])
            if (QD[cx, cy] > TH):
                bits += str(1)
                TH = alpha * TH + (1-alpha) * QD[cx, cy]
            else:
                bits += str(0) 
                TH = alpha * TH + (1-alpha) * QD[cx, cy]

            # bits += str(int(QD[cx, cy]) & 1)
            collected += 1

        if collected >= msg_bits:
            break

    # Convert to string
    out = ""
    for i in range(0, len(bits), 8):
        out += chr(int(bits[i:i+8], 2))

    return out

def calculate_ber(original_message, decoded_message):
    """
    Calculates the Bit Error Rate (BER) between the original and decoded messages.
    """
    if not original_message or not decoded_message:
        print("Error: Both original and decoded messages must be non-empty.")
        return None

    # 1. Convert messages to full binary strings (8 bits per character)
    
    # Original (Ground Truth) Bitstream
    original_bits = ''.join(format(ord(c), '08b') for c in original_message)
    
    # Decoded Bitstream (Extracted Result)
    decoded_bits = ''.join(format(ord(c), '08b') for c in decoded_message)
    
    # 2. Match the lengths (If they differ, it means the message was truncated/corrupted)
    # We only compare up to the length of the shorter stream.
    min_length = min(len(original_bits), len(decoded_bits))
    
    original_bits = original_bits[:min_length]
    decoded_bits = decoded_bits[:min_length]
    
    total_bits = min_length
    error_count = 0

    # 3. Compare bit-by-bit
    for bit_orig, bit_dec in zip(original_bits, decoded_bits):
        if bit_orig != bit_dec:
            error_count += 1
            
    # 4. Calculate BER
    if total_bits == 0:
        return 0.0

    ber = error_count / total_bits
    
    return ber

def calculate_metrics(img1, img2):
    # Ensure images are in the same range (0-255) and type (uint8)
    # If your input is 0-1 float, convert to 0-255 uint8
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.max() <= 1.0:
        img2 = (img2 * 255).astype(np.uint8)

    psnr_value = psnr(img1, img2)
    
    print(f"PSNR: {psnr_value:.2f} dB")

    return psnr_value
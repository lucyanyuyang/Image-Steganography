import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import cv2

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

def idct2(a): return idct(idct(a.T, norm='ortho').T, norm='ortho')

def encode_dct(img, message, out_path, beta):

    img_uint8 = (img * 255).astype(np.uint8)

    y_crcb_np = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
    
    # Split the channels (Y, Cr, Cb) as NumPy arrays
    Y = y_crcb_np[:, :, 0].astype(np.float32) # Luminance channel
    Cb = y_crcb_np[:, :, 2].astype(np.uint8)  # Blue-difference chrominance
    Cr = y_crcb_np[:, :, 1].astype(np.uint8)  # Red-difference chrominance
    
    rows, cols = Y.shape

    # Convert message to bits
    bits = '11110000'
    bits += format(len(message), '016b')
    bits += ''.join(format(ord(c), '08b') for c in message)
    total = len(bits)

    capacity = (rows//8)*(cols//8)
    if total > capacity:
        raise ValueError("Message too large")

    idx = 0
    cx, cy = 3, 3  # embedding coefficient

    for i in range(0, rows-7, 8):
        for j in range(0, cols-7, 8):
            if idx >= total:
                break

            block = Y[i:i+8, j:j+8] - 128
            D = dct2(block)

            # quantize
            QD = np.round(D / Q)

            # embed bit
            coeff = int(QD[cx, cy])
            bit = int(bits[idx])
            
            
            if bit == 0:
                QD[cx, cy] = -beta
            else:
                QD[cx, cy] = beta
            idx += 1

            # dequantize
            D2 = QD * Q
            Y[i:i+8, j:j+8] = np.clip(idct2(D2) + 128, 0, 255)

        if idx >= total:
            break

    # Recombine YCbCr → RGB
    final_YCrCb = np.stack((Y.astype(np.uint8), Cr, Cb), axis=-1)
    final_rgb_np = cv2.cvtColor(final_YCrCb, cv2.COLOR_YCrCb2RGB)
    
    # Convert back to PIL Image only for saving
    final = Image.fromarray(final_rgb_np)
    final.save(out_path)

    print("Message embedded successfully.")
import sys
from PIL import Image

import numpy as np


def MSE(img1: np.ndarray, img2: np.ndarray, margin: int = 0) -> float | float:
    """
    Computes mean square error and peak signal to noise ratio of two 
    gray images.
    """

    m, n = img1.shape
    if margin != 0:
        img1 = img1[margin + 1:m-margin, margin+1:n - margin]
        img2 = img2[margin + 1:m-margin, margin+1:n - margin]

    img1 = img1.astype(np.double)
    img2 = img2.astype(np.double)

    error = img1 - img2
    mse = np.multiply(error, error).flatten().sum() / (m * n)
    psnr = 10 * np.log10(255 ** 2 / mse)

    return mse, psnr

def main():
    img1 = np.asarray(Image.open(sys.argv[1]).convert('L'))
    img2 = np.asarray(Image.open(sys.argv[2]).convert('L'))

    mse, psnr = MSE(img1, img2)
    print(f'MSE: {mse:.0f}\nPSNR: {psnr:.4f}')

if __name__ == '__main__':
    main()
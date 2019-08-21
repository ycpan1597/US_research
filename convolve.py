from skimage.exposure import rescale_intensity
import numpy as np
import cv2

def convolve(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2] # for most cases, kH and kW should be the same

    # padding the input image ensures that the output has the same dimension as the inputs
    pad = (kH - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype = 'float32')

    # slides the kernal one pixel at a time to generate the convolved result
    for y in np.arange(pad, pad + iH):
        for x in np.arange(pad, pad + iW):
            roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output
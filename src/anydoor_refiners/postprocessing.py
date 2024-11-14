import cv2
import numpy as np


def post_processing(
    predicted_image: np.ndarray,
    background_image: np.ndarray,
    sizes: np.ndarray,
    background_box: np.ndarray,
):
    """Crop back"""

    H1, W1, H2, W2 = sizes
    y1, y2, x1, x2 = background_box
    predicted_image = cv2.resize(predicted_image, (W2, H2))
    m = 5  # maigin_pixel

    if W1 == H1:
        background_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = predicted_image[
            m:-m, m:-m
        ]
        return background_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        predicted_image = predicted_image[:, pad1:-pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        predicted_image = predicted_image[pad1:-pad2, :, :]

    gen_image = background_image.copy()
    gen_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = predicted_image[m:-m, m:-m]
    return gen_image

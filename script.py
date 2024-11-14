import cv2
import numpy as np
from PIL import Image


def turn_to_mask(image_path, output_path="output_mask.png"):

    tar_mask = Image.open(image_path ).convert('P')
    tar_mask= np.array(tar_mask)
    tar_mask = tar_mask == 5
    tar_mask = tar_mask.astype(np.uint8) * 255
    cv2.imwrite(output_path, tar_mask)
    print(f"Mask saved to {output_path}")

def turn_to_black(image_path, output_path="output_mask.png"):

    tar_mask = Image.open(image_path)
    shapes = np.array(tar_mask).shape
    black_image = np.zeros(shapes)
    cv2.imwrite(output_path, black_image)
    print(f"Mask saved to {output_path}")


def save_image_with_alpha(image_path, mask_path, output_path="output_with_alpha.png"):
    """
    Takes an image path and a mask path, and saves the image with an alpha channel 
    (transparency) based on the mask.
    
    Args:
    - image_path (str): Path to the input image.
    - mask_path (str): Path to the mask image. Mask should be a single-channel (grayscale) image.
    - output_path (str): Path to save the output image with alpha channel. Default is 'output_with_alpha.png'.
    """
    
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the mask image (assumed to be a grayscale or binary image)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Mask not found or path is incorrect")

    # Ensure the mask has the same dimensions as the image
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("Mask dimensions must match the image dimensions")
    
    # Create the alpha channel based on the mask
    # Normalize the mask to ensure alpha values range from 0 to 255
    alpha_channel = np.where(mask > 128, 255, 0).astype(np.uint8)
    
    # Add the alpha channel to the original image to make it RGBA
    rgba_image = cv2.merge([image, alpha_channel])

    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)

    # Save the image with alpha channel in PNG format
    cv2.imwrite(output_path, rgba_image)
    print(f"Image with alpha channel saved to {output_path}")

# Example usage:
turn_to_black("examples/backgrounds/5.png", "examples/background_masks/5.png")
turn_to_black("examples/backgrounds/6.png", "examples/background_masks/6.png")

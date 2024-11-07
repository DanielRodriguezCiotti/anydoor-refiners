import numpy as np
import cv2
from typing import Tuple, List, Dict


def box2square(
    image: np.ndarray, box: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Convert a bounding box to a square by expanding along its shortest dimension."""
    H, W = image.shape[:2]
    y1, y2, x1, x2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = y2 - y1, x2 - x1

    # Make square by adjusting shorter side
    if h >= w:
        x1, x2 = cx - h // 2, cx + h // 2
    else:
        y1, y2 = cy - w // 2, cy + w // 2

    # Clip to image boundaries
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    return y1, y2, x1, x2


def box_in_box(
    small_box: Tuple[int, int, int, int], big_box: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Adjust a smaller box to be relative to a larger bounding box's origin."""
    y1, y2, x1, x2 = small_box
    y1_b, _, x1_b, _ = big_box
    return y1 - y1_b, y2 - y1_b, x1 - x1_b, x2 - x1_b


def expand_bbox(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ratio: List[float] = [1.2, 2.0],
    min_crop: int = 0,
) -> Tuple[int, int, int, int]:
    """Expand a bounding box by a random ratio within given bounds."""
    y1, y2, x1, x2 = bbox
    # Calculate random ratio and store it in a new variable
    expansion_ratio = np.random.randint(int(ratio[0] * 10), int(ratio[1] * 10)) / 10.0
    H, W = mask.shape[:2]
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = max(expansion_ratio * (y2 - y1 + 1), min_crop), max(
        expansion_ratio * (x2 - x1 + 1), min_crop
    )

    # Compute expanded coordinates, ensuring they stay within image bounds
    x1, x2 = int(xc - w / 2), int(xc + w / 2)
    y1, y2 = int(yc - h / 2), int(yc + h / 2)
    return max(0, y1), min(H, y2), max(0, x1), min(W, x2)


def pad_to_square(
    image: np.ndarray, pad_value: int = 255, random_pad: bool = False
) -> np.ndarray:
    """Pad an image to make it square."""
    H, W = image.shape[:2]
    if H == W:
        return image

    padd = abs(H - W)
    pad_1 = np.random.randint(0, padd) if random_pad else padd // 2
    pad_2 = padd - pad_1

    if H > W:
        pad_param = ((0, 0), (pad_1, pad_2), (0, 0))
    else:
        pad_param = ((pad_1, pad_2), (0, 0), (0, 0))

    return np.pad(image, pad_param, "constant", constant_values=pad_value)


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Retrieve the bounding box from a binary mask."""
    h, w = mask.shape[:2]
    if mask.sum() < 10:
        return 0, h, 0, w

    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return y1, y2, x1, x2


def expand_image_mask(
    image: np.ndarray, mask: np.ndarray, ratio: float = 1.4
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand an image and mask by a scaling ratio, maintaining content centered."""
    h, w = image.shape[:2]
    H, W = int(h * ratio), int(w * ratio)

    # Calculate padding for height and width
    h1 = (H - h) // 2
    h2 = H - h - h1
    w1 = (W - w) // 2
    w2 = W - w - w1

    # Pad the image and mask, maintaining original content centered
    pad_image = ((h1, h2), (w1, w2), (0, 0))  # Padding for color image
    pad_mask = ((h1, h2), (w1, w2))  # Padding for single-channel mask

    # Apply padding
    expanded_image = np.pad(image, pad_image, mode="constant", constant_values=255)
    expanded_mask = np.pad(mask, pad_mask, mode="constant", constant_values=0)

    return expanded_image, expanded_mask


def sobel(img: np.ndarray, mask: np.ndarray, thresh: int = 50) -> np.ndarray:
    """Apply Sobel edge detection and mask high-frequency content."""
    '''Calculating the high-frequency map.'''
    H,W = img.shape[0], img.shape[1]
    img = cv2.resize(img,(256,256))
    mask = (cv2.resize(mask,(256,256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)
    
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
    sobel_Y = cv2.convertScaleAbs(sobely)
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1) * mask    
    
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr,scharr,scharr],-1)
    scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
    scharr = cv2.resize(scharr,(W,H))
    return scharr


def preprocess_images(
    object_image: np.ndarray,
    object_mask: np.ndarray,
    background_image: np.ndarray,
    background_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Preprocess and augment images for compositing."""
    # ========== Object Processing ==========
    # Get object bounding box and mask it
    obj_bbox = get_bbox_from_mask(object_mask)
    obj_mask_3ch = np.stack([object_mask] * 3, axis=-1)
    masked_object_image = object_image * obj_mask_3ch + 255 * (1 - obj_mask_3ch)
    y1, y2, x1, x2 = obj_bbox
    cropped_object_image = masked_object_image[y1:y2, x1:x2]
    cropped_object_mask = object_mask[y1:y2, x1:x2]

    # Expand, pad, and resize object image
    scale_ratio = np.random.uniform(1.2, 1.3)
    expanded_object_image, expanded_object_mask = expand_image_mask(
        cropped_object_image, cropped_object_mask, ratio=scale_ratio
    )
    expanded_object_mask_3ch = np.stack([expanded_object_mask] * 3, axis=-1)
    padded_object_image = cv2.resize(
        pad_to_square(expanded_object_image), (224, 224)
    ).astype(np.uint8)
    padded_object_mask_3ch = cv2.resize(
        pad_to_square(expanded_object_mask_3ch * 255, pad_value=0), (224, 224)
    ).astype(np.uint8)

    # Sobel-filtered collage image
    collage_bg_image = sobel(padded_object_image, padded_object_mask_3ch[:, :, 0]  / 255)

    # ========== Background Processing ==========
    # Get background bounding box and expand
    background_bbox = get_bbox_from_mask(background_mask)
    expanded_background_bbox = expand_bbox(
        background_mask, background_bbox, ratio=[1.1, 1.2]
    )
    # Crop and expand the background bounding box, then make it square
    cropped_background_bbox = box2square(
        background_image,
        expand_bbox(background_image, expanded_background_bbox, ratio=[1.5, 3]),
    )
    y1, y2, x1, x2 = cropped_background_bbox
    cropped_background_image = background_image[y1:y2, x1:x2]

    # Determine the target collage box within the background
    target_collage_box = box_in_box(expanded_background_bbox, cropped_background_bbox)
    y1, y2, x1, x2 = target_collage_box

    # Resize reference collage and mask to fit in the background bounding box
    ref_collage_resized = cv2.resize(collage_bg_image, (x2 - x1, y2 - y1))
    ref_mask_resized = cv2.resize(padded_object_mask_3ch, (x2 - x1, y2 - y1)).astype(
        np.uint8
    )
    ref_mask_resized = (ref_mask_resized > 128).astype(np.uint8)

    # ========== Compositing ==========
    # Place the object collage onto the background
    collage = cropped_background_image.copy()
    collage[y1:y2, x1:x2, :] = ref_collage_resized

    collage_mask = np.zeros_like(cropped_background_image)
    collage_mask[y1:y2, x1:x2, :] = 1.0

    # Sizes before and after padding
    padded_background_image = pad_to_square(cropped_background_image, pad_value=0).astype(np.uint8)
    padded_collage = pad_to_square(collage, pad_value=0).astype(np.uint8)
    padded_collage_mask = pad_to_square(collage_mask, pad_value=-1).astype(np.uint8)

    # Resize to final output size
    final_bg_image = cv2.resize(padded_background_image, (512, 512)).astype(np.float32)
    final_collage = cv2.resize(padded_collage, (512, 512)).astype(np.float32)
    final_collage_mask = (
        cv2.resize(padded_collage_mask, (512, 512)).astype(np.float32) > 0.5
    ).astype(np.float32)

    # Normalize images
    masked_object_image_aug = padded_object_image / 255.0
    final_bg_image = final_bg_image / 127.5 - 1.0
    final_collage = final_collage / 127.5 - 1.0
    final_collage = np.concatenate(
        [final_collage, final_collage_mask[:, :, :1]], axis=-1
    )

    # Package results
    item = dict(
        object=masked_object_image_aug.copy(),
        background=final_bg_image.copy(),
        collage=final_collage.copy(),
        tar_box_yyxx_crop=np.array(cropped_background_bbox),
    )

    return item

import numpy as np
from scipy.ndimage import rotate
import cv2
from PIL import Image as im
from typing import Tuple

def find_score(arr, angle: int) -> Tuple[np.ndarray, float]:
    """
    Find the score of the skew angle to be used in deskewing the image
    
    Args:
    arr: the image array
    angle: the angle to rotate the image by
    
    Returns:
    hist: the histogram of the image
    score: the score of the skew angle
    """
    
    # mode{‘reflect’, ‘grid-mirror’, ‘constant’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’}
    data = rotate(arr, angle, reshape=False, order=0, mode='constant', cval=0, prefilter=False)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def rotate_image(image, angle: int):
    """
    Rotates an image by a given angle and fills the remaining pixels with white color.

    Args:
        image: A NumPy array representing the input image.
        angle: The rotation angle in degrees.

    Returns:
        A new NumPy array representing the rotated image.
    """
    # Get image height and width
    height, width = image.shape[:2]

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Perform the rotation and fill the remaining pixels with white color
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated_image

def deskew(binary_img) -> np.ndarray:
    """
    Deskew the image
    
    Args:
    binary_img: the binary image
    
    Returns:
    pix: the deskewed image
    """
    bin_img = (binary_img // 255.0)

    angles = np.array ([0 , 45 , 90 , 135 , 180 , 225 , 270 , 315])
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    # correct skew
    data = rotate_image(bin_img, best_angle)
    img = im.fromarray((255 * data).astype("uint8"))

    pix = np.array(img)
    return pix

def preprocess(img):
    """
    Preprocess the image
    
    Args:
    img: the image
    
    Returns:
    img: the preprocessed image
    """
    image_size = 600
    sharpen_kernel = np.array([[0,-1, 0], [-1,5,-1], [0,-1,0]])
    img = cv2.medianBlur(img, 3) # To remove Salt and Pepper noise
    img = cv2.filter2D(img, -1, sharpen_kernel)  # Sharpen the image
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # Convert the image to binary
    img = cv2.bitwise_not(img) if np.mean(img) > 127 else img # Invert the image if the mean is less than 127 
    img = deskew(img) # Deskew the image
    final_img = cv2.resize(img, (image_size, image_size)) # Resize the image
    return final_img

def remove_rows(image, threshold=0.008):
    """
    Remove rows with white pixels less than 10% of the image size
    
    Args:
    X_preprocess_sliced: the list of images
    max_shape_index: the index of the image to process
    
    Returns:
    image: the image after removing the rows
    """
    white_pixels_per_row = np.sum(image == 255, axis=1)
    rows_to_remove = white_pixels_per_row < image.shape[1] * threshold
    image = image[~rows_to_remove]
    return image

def remove_columns(image, threshold = 0.1):
    """
    Remove columns with white pixels less than 10% of the image size
    
    Args:
    X_preprocess_sliced: the list of images
    max_shape_index: the index of the image to process
    
    Returns:
    image: the image after removing the columns
    """
    white_pixels_per_column = np.sum(image == 255, axis=0)
    # white < 0.1 * height
    columns_to_remove = white_pixels_per_column < image.shape[0] * threshold
    image = image[:, ~columns_to_remove]
    return image

def pad_image(image):
    """
    Pad the image with zeros if the width is less than 515 and the height is less than 270
    
    Args:
    image: the image
    
    Returns:
    image: the padded image
    """
    if image.shape[1] < 515:
        pad_width = 515 - image.shape[1]
        image = np.pad(image, ((0, 0), (0, pad_width)), 'constant', constant_values=(0, 0))
    if image.shape[0] < 270:
        pad_height = 270 - image.shape[0]
        image = np.pad(image, ((0, pad_height), (0, 0)), 'constant', constant_values=(0, 0))
    return image

def crop_image(image):
    """
    Crop the image if the width is more than 515 and the height is more than 270
    
    Args:
    image: the image
    
    Returns:
    image: the cropped image
    """
    if image.shape[1] > 515:
        crop_width = image.shape[1] - 515
        image = image[:, crop_width//2:-(crop_width//2)]
    if image.shape[0] > 267:
        crop_height = image.shape[0] - 267
        image = image[crop_height//2:-(crop_height//2), :]
    return cv2.resize(image, (515, 270))

def preprocess_new(img, columns_threshold=0.001, rows_threshold=0.005):
    """
    Preprocess the image
    
    Args:
    img: the image
    
    Returns:
    img: the preprocessed image
    """
    img = remove_columns(img, columns_threshold)
    img = remove_rows(img, rows_threshold)
    img = pad_image(img)
    final_img = crop_image(img)
    return final_img
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
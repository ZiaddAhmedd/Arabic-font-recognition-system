import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter
from PIL import Image as im
import pickle
import cv2
from tqdm import tqdm
from scipy.ndimage import rotate
import time
import joblib
from sklearn.metrics import accuracy_score,f1_score,classification_report
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from skimage.feature import hog
from sklearn.calibration import LabelEncoder
from copy import deepcopy

def show_images(images,titles=None):
    """
    This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    images[0] will be drawn with the title titles[0] if exists.
    """
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 

def load_images():
    """
    This function is used to load the images from the fonts-dataset folder.
    """
    images_train = []
    labels_train = []
    filenames = []
    labels = ['IBM Plex Sans Arabic', 'Lemonada', 'Marhey', 'Scheherazade New']
    empty_images_filenames = ["360.jpeg","627.jpeg","853.jpeg"] 
    # Use tqdm to show a progress bar
    for i in tqdm(labels):
        for filename in os.listdir(f'fonts-dataset/{i}'):
            img = cv2.imread(f'fonts-dataset/{i}/{filename}', cv2.IMREAD_GRAYSCALE)
            if i == "Lemonada" and filename in empty_images_filenames:
                print(f"{filename} is empty image!")
                continue
            images_train.append(img)
            labels_train.append(i)
            filenames.append(filename)
    return images_train, labels_train, filenames

def find_score(arr, angle):
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

def rotate_image(image, angle):
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
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))

    return rotated_image

def deskew(binary_img):
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
    deskewed_img = deskew(img) # Deskew the image
    final_img = cv2.bitwise_not(deskewed_img) if np.mean(deskewed_img) > 127 else deskewed_img # Invert the image if the mean is less than 127 
    final_img = cv2.resize(final_img, (image_size, image_size)) # Resize the image
    return final_img

def apply_hog(X_train_preprocess):
    X_train_hog = []
    for i in tqdm(X_train_preprocess):
        X_train_hog.append(hog(i, orientations= 16, pixels_per_cell=(32, 32), cells_per_block=(4, 4), block_norm='L2-Hys'))
    X_train_hog = np.array(X_train_hog)
    return X_train_hog

def apply_sift(X_train_preprocess):
    sift = cv2.SIFT_create()

    X_train_sift = []
    for i in tqdm(X_train_preprocess):
        kp, des = sift.detectAndCompute(i, None)
        if des is None:
            # Add a row of zeros to the SIFT descriptors
            des = np.zeros((1, 128))
        des = des.flatten()
        X_train_sift.append(des)
    return X_train_sift
    
# Pad the SIFT descriptors to the maximum length
def pad_sift_descriptors(X_train_sift, fixed_len):
    # Create a generator that yields each padded descriptor on-the-fly
    padded_descriptors = (np.pad(des, (0, max(0, fixed_len - des.shape[0])))[:fixed_len] for des in X_train_sift)

    # Convert the generator to a numpy array
    X_train_sift_np = np.array(list(padded_descriptors))
    return X_train_sift_np

def Laplacian_filter(img):
    laplacian_filter = np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]
    ])
    edge_image = convolve2d(img, laplacian_filter)
    edge_image = np.where(edge_image > 0.5, edge_image, 0)
    edge_image = np.where(edge_image < 0.5, edge_image, 255)

    edge_image = 255 - edge_image
    return edge_image

def get_edm1_matrix(edge_image):
    edm_matrix = np.zeros((3,3))
    # Pad the image with ones
    edge_image = np.pad(edge_image, 1, mode='constant', constant_values=1)
    for i in range(0, edge_image.shape[0]):
        for j in range(0, edge_image.shape[1]):
            if edge_image[i, j] == 0:
                edm_matrix[1,1] += 1
                if edge_image[i, j + 1] == 0:
                    edm_matrix[1,2] += 1
                if edge_image[i + 1, j + 1] == 0:
                    edm_matrix[2,2] += 1
                if edge_image[i + 1, j] == 0:
                    edm_matrix[2,1] += 1
                if edge_image[i + 1, j - 1] == 0:
                    edm_matrix[2,0] += 1
                if edge_image[i, j - 1] == 0:
                    edm_matrix[1,0] += 1
                if edge_image[i - 1, j - 1] == 0:
                    edm_matrix[0,0] += 1
                if edge_image[i - 1, j] == 0:
                    edm_matrix[0,1] += 1
                if edge_image[i - 1, j + 1] == 0:
                    edm_matrix[0,2] += 1
    return edm_matrix

def sort_values(edm1_matrix):
    values = edm1_matrix.flatten()
    edm1_occurrences_sorted = {}
    edm1_occurrences_sorted[values[5]] = [5,3]
    if values[2] not in edm1_occurrences_sorted:
        edm1_occurrences_sorted[values[2]] = [2,6]
    else:
        edm1_occurrences_sorted[values[2]].extend([2,6])
    if values[1] not in edm1_occurrences_sorted:
        edm1_occurrences_sorted[values[1]] = [1,7]
    else:
        edm1_occurrences_sorted[values[1]].extend([1,7])
    if values[0] not in edm1_occurrences_sorted:
        edm1_occurrences_sorted[values[0]] = [0,8]
    else:
        edm1_occurrences_sorted[values[0]].extend([0,8])
    edm1_occurrences_sorted = dict(sorted(edm1_occurrences_sorted.items(), reverse=True))

    lst = []
    for key in edm1_occurrences_sorted:
        lst.extend(edm1_occurrences_sorted[key])
    return lst

def get_first_occurrence(neighboring_indices, edm1_occurrences_sorted_list):
    for idx in edm1_occurrences_sorted_list:
        if idx in neighboring_indices:
            return idx
        
def get_edm2_matrix(edge_image, edm1_matrix):
    edm2_matrix_flattened = np.zeros(9)
    edm1_occurrences_sorted_list = sort_values(edm1_matrix)
    edm2_matrix_flattened[4] = edm1_matrix[1,1]
    edge_image = np.pad(edge_image, 1, mode='constant', constant_values=1)
    for i in range(0, edge_image.shape[0]):
        for j in range(0, edge_image.shape[1]):
            neighboring_indices = []
            if edge_image[i, j] == 0:
                if edge_image[i, j + 1] == 0:
                    neighboring_indices.append(5)
                if edge_image[i - 1, j + 1] == 0:
                    neighboring_indices.append(2)
                if edge_image[i - 1, j] == 0:
                    neighboring_indices.append(1)
                if edge_image[i - 1, j - 1] == 0:
                    neighboring_indices.append(0)
                if edge_image[i, j - 1] == 0:
                    neighboring_indices.append(3)
                if edge_image[i + 1, j - 1] == 0:
                    neighboring_indices.append(6)
                if edge_image[i + 1, j] == 0:  
                    neighboring_indices.append(7)
                if edge_image[i + 1, j + 1] == 0:
                    neighboring_indices.append(8)

                first_occurrence = get_first_occurrence(neighboring_indices, edm1_occurrences_sorted_list)
                edm2_matrix_flattened[first_occurrence] += 1
    edm2_matrix = edm2_matrix_flattened.reshape(3,3)   
    return edm2_matrix

def apply_edm(X_train_preprocess):
    edge_images = [Laplacian_filter(img) for img in tqdm(X_train_preprocess)]
    
    edm1_matrices = [get_edm1_matrix(edge_img) for edge_img in tqdm(edge_images)]
    edm2_matrices = [get_edm2_matrix(edge_images[i], edm1_matrices[i]) for i in tqdm(range(len(edge_images)))]
    
    edm1_matrices = np.array(edm1_matrices)
    edm2_matrices = np.array(edm2_matrices)

    edm1_matrices = edm1_matrices.reshape(-1,9)
    edm2_matrices = edm2_matrices.reshape(-1,9)
    return edm1_matrices, edm2_matrices

def apply_additional_edm_features(X_edm1, X_edm2):
    edges_direction = np.max(X_edm1, axis=1)
    edges_direction = edges_direction.reshape(-1,1)
    homogeneity = np.array([x/np.sum(x) for x in X_edm1])
    pixel_regularity = np.array([x/x[4] for x in X_edm1])
    edges_regularity = np.array([x / x[4] for x in X_edm2])
    return edges_direction, homogeneity, pixel_regularity, edges_regularity

class Preprocessing():
    def __init__(self, preprocess_pipe):
        self.preprocess_pipe = preprocess_pipe
        
    def preprocess_data(self, X, test=False):
        fixed_len = 128 * 350
        X_preprocess = [preprocess(i) for i in tqdm(X)]
        X_preprocess = np.array(X_preprocess)
        X_hog = apply_hog(X_preprocess)
        X_sift = apply_sift(X_preprocess)
        X_sift_padded = pad_sift_descriptors(X_sift, fixed_len)
        X_features = np.concatenate((X_hog, X_sift_padded), axis=1)
        if test:
            X_features_transformed = self.preprocess_pipe.transform(X_features)
        else:
            X_features_transformed = self.preprocess_pipe.fit_transform(X_features)
            
        with open('preprocess_pipe.pkl', 'wb') as f:
            pickle.dump(self.preprocess_pipe, f)
            
        return X_features_transformed
    
class PyTorchClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, preprocess_pipe, learning_rate=0.0002 , epoch=50):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.best_accuracy = -1  # Initialize with a value that will definitely be improved upon
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.preprocess_pipe = preprocess_pipe
        self.model = self.create_model()

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.output_dim),
        )
        return model
        
    
    def fit(self, X_train_features, X_val_features, y_train_labels, y_val_labels, labels):
        y_train =  [labels.index(i) for i in y_train_labels]
        y_val = [labels.index(i) for i in y_val_labels]
        
        lb = LabelBinarizer()
        y_train_one_hot = lb.fit_transform(y_train)
        y_val_one_hot = lb.fit_transform(y_val)
        
        X_train_tensor = torch.FloatTensor(X_train_features)
        y_train_tensor = torch.LongTensor(y_train_one_hot)

        X_val_tensor = torch.FloatTensor(X_val_features)
        y_val_tensor = torch.LongTensor(y_val_one_hot)
        
        # Create a dataset
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create a dataloader
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create a tqdm object
        progress_bar = tqdm(range(self.epoch), desc="Epoch", leave=False)

        for self.epoch in progress_bar:
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, torch.max(y_batch, 1)[1])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Calculate accuracy on the validation set
            val_acc = 0
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    outputs = self.model(X_val)
                    _, predicted = torch.max(outputs, 1)
                    val_acc += (predicted == torch.max(y_val, 1)[1]).sum().item()
            accuracy = val_acc / len(y_val_tensor)

            # If the current model has better accuracy, save the model parameters
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model_state = deepcopy(self.model.state_dict(prefix="model."))

            # Update the progress bar
            progress_bar.set_postfix({'Loss': f'{total_loss:.4f}', 'Accuracy': f'{self.best_accuracy:.4f}'})
            
        # Save the best model parameters to the model
        self.save_best_model('best_model.pth')

    def predict(self, X):
        preprocess_module = Preprocessing(self.preprocess_pipe)
        X_features_transformed = preprocess_module.preprocess_data(X, test=True)
        X_tensor = torch.FloatTensor(X_features_transformed)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        _, predicted = torch.max(predictions, 1)
        return predicted.numpy()

    def save_best_model(self, filepath):
        torch.save(self.best_model_state, filepath)
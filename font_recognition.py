import numpy as np
import pickle
import cv2
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.signal import convolve2d
import torch
import torch.nn as nn
import torch.optim
from typing import List
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelBinarizer
from skimage.feature import hog
from copy import deepcopy
from utils import *
from preprocess import *

def apply_hog(X_train_preprocess: List[np.ndarray]) -> np.ndarray:
    X_train_hog = []
    for i in X_train_preprocess:
        X_train_hog.append(hog(i, orientations= 16, pixels_per_cell=(32, 32), cells_per_block=(4, 4), block_norm='L2-Hys'))
    X_train_hog = np.array(X_train_hog)
    return X_train_hog

def apply_sift(X_train_preprocess: List[np.ndarray]) -> List[np.ndarray]:
    sift = cv2.SIFT_create()

    X_train_sift = []
    for i in X_train_preprocess:
        kp, des = sift.detectAndCompute(i, None)
        if des is None:
            # Add a row of zeros to the SIFT descriptors
            des = np.zeros((1, 128))
        des = des.flatten()
        X_train_sift.append(des)
    return X_train_sift
    
# Pad the SIFT descriptors to the maximum length
def pad_sift_descriptors(X_train_sift: List[np.ndarray], fixed_len: int) -> np.ndarray:
    padded_descriptors = (np.pad(des, (0, max(0, fixed_len - des.shape[0])))[:fixed_len] for des in X_train_sift)
    X_train_sift_np = np.array(list(padded_descriptors))
    return X_train_sift_np

def evaluate(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.4f}%')
    
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

class Preprocessing():
    def __init__(self, preprocess_pipe):
        self.preprocess_pipe = preprocess_pipe
        
    def preprocess_data(self, X, test: bool = False) -> np.ndarray:
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
    
    def preprocess_test_data(self, X) -> np.ndarray:
        fixed_len = 128 * 350
        X_preprocess = preprocess(X)
        X_preprocess = [np.array(X_preprocess)]
        X_hog = apply_hog(X_preprocess)
        X_sift = apply_sift(X_preprocess)
        X_sift_padded = pad_sift_descriptors(X_sift, fixed_len)
        X_features = np.concatenate((X_hog, X_sift_padded), axis=1)
        X_features_transformed = self.preprocess_pipe.transform(X_features)
        return X_features_transformed
 
    def preprocess_test(self, X):
        fixed_len = 128 * 350
        X = preprocess(X)
        X_preprocess = [np.array(X)]
        X_hog = apply_hog(X_preprocess)
        X_sift = apply_sift(X_preprocess)
        X_sift_padded = pad_sift_descriptors(X_sift, fixed_len)
        X_features = np.concatenate((X_hog, X_sift_padded), axis=1)
        X_features_transformed = self.preprocess_pipe.transform(X_features)
        X_preprocess_edm = preprocess_new(X)
        X_laplace = Laplacian_filter(X_preprocess_edm)
        edm1_matrix = get_edm1_matrix(X_laplace)
        edm1_matrix = np.array(edm1_matrix)
        edm1_matrix = edm1_matrix.reshape(-1,9)
        edm2_matrix = get_edm2_matrix(X_laplace, edm1_matrix.reshape(3,3))
        edm2_matrix = np.array(edm2_matrix)
        edm2_matrix = edm2_matrix.reshape(-1,9)
        X_features_edm = np.concatenate((edm1_matrix, edm2_matrix), axis=1)
        return X_features_transformed, X_features_edm

class PyTorchClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, learning_rate=0.0002 , epoch=50):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.best_accuracy = -1  # Initialize with a value that will definitely be improved upon
        self.learning_rate = learning_rate
        self.epoch = epoch
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
        X_tensor = torch.FloatTensor(X).to("cpu")
        with torch.no_grad():
            predictions = self.model(X_tensor)
            # Apply softmax to the predictions
            predictions = nn.functional.softmax(predictions, dim=1)
        return predictions.cpu().numpy()

    def save_best_model(self, filepath):
        torch.save(self.best_model_state, filepath)
        
if __name__ == "__main__":
    labels = ['Scheherazade New', 'Marhey', 'Lemonada', 'IBM Plex Sans Arabic']

    preprocess_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.99)),
    ])
    
    X_data, y_labels, _ = load_images()
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels, test_size=0.20, random_state=42, stratify=y_labels)
    
    #########################################################################################
    # Train the model
    #########################################################################################
    preprocess_module = Preprocessing(preprocess_pipe)
    X_train_features = preprocess_module.preprocess_data(X_train)
    input_dim = X_train_features.shape[1]
    X_val_features = preprocess_module.preprocess_data(X_val, test=True)
    pytorch_model = PyTorchClassifier(input_dim, 512, 256, len(labels), learning_rate=0.00025, epoch=50)
    pytorch_model.fit(X_train_features, X_val_features, y_train, y_val, labels)


    #########################################################################################
    # Test the model
    #########################################################################################
    
    X_test = []
    y_test = []
    for i in tqdm(labels):
        for filename in os.listdir(f'content/test/{i}'):
            img = cv2.imread(f'content/test/{i}/{filename}', cv2.IMREAD_GRAYSCALE)
            X_test.append(img)
            y_test.append(i)
            
    y_test =  [labels.index(i) for i in y_test]
        
    preprocess_module = Preprocessing(preprocess_pipe)
    X_test_features_transformed = preprocess_module.preprocess_data(X_test, test=True)
    y_pred = pytorch_model.predict(X_test_features_transformed)
    accuracy = evaluate(y_pred, y_test)
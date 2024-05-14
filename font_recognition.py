import numpy as np
import pickle
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelBinarizer
from skimage.feature import hog
from copy import deepcopy
from utils import *
from preprocess import *

def apply_hog(X_train_preprocess):
    X_train_hog = []
    for i in X_train_preprocess:
        X_train_hog.append(hog(i, orientations= 16, pixels_per_cell=(32, 32), cells_per_block=(4, 4), block_norm='L2-Hys'))
    X_train_hog = np.array(X_train_hog)
    return X_train_hog

def apply_sift(X_train_preprocess):
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
def pad_sift_descriptors(X_train_sift, fixed_len):
    # Create a generator that yields each padded descriptor on-the-fly
    padded_descriptors = (np.pad(des, (0, max(0, fixed_len - des.shape[0])))[:fixed_len] for des in X_train_sift)

    # Convert the generator to a numpy array
    X_train_sift_np = np.array(list(padded_descriptors))
    return X_train_sift_np

def evaluate(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.4f}%')

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
    
    def preprocess_test_data(self, X):
        fixed_len = 128 * 350
        X_preprocess = preprocess(X)
        X_preprocess = [np.array(X_preprocess)]
        X_hog = apply_hog(X_preprocess)
        X_sift = apply_sift(X_preprocess)
        X_sift_padded = pad_sift_descriptors(X_sift, fixed_len)
        X_features = np.concatenate((X_hog, X_sift_padded), axis=1)
        X_features_transformed = self.preprocess_pipe.transform(X_features)
        return X_features_transformed

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
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        _, predicted = torch.max(predictions, 1)
        return predicted.numpy()

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
        for filename in os.listdir(f'content/train/{i}'):
            img = cv2.imread(f'content/train/{i}/{filename}', cv2.IMREAD_GRAYSCALE)
            X_test.append(img)
            y_test.append(i)
            
    y_test =  [labels.index(i) for i in y_test]
        
    preprocess_module = Preprocessing(preprocess_pipe)
    X_test_features_transformed = preprocess_module.preprocess_data(X_test, test=True)
    y_pred = pytorch_model.predict(X_test_features_transformed)
    accuracy = evaluate(y_pred, y_test)
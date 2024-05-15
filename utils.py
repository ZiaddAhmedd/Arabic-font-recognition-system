import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import os
from typing import List, Optional

def show_images(images, titles: Optional[List[str]] = None) -> None:
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
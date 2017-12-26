import argparse
import tensorflow as tf
import scipy.misc
import numpy as np
import pandas as pd
import cv2

def preprocess_prediction(img):
    # img = img.reshape(48,48)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48,48))
    img = img.reshape(-1)
    img = img - img.mean()
    img = np.multiply(img,1/255.0)
    img = img.astype(np.float32)
    return img

def preprocessing(path="/Users/sharansankar/OneDrive/Documents/data_science_resources/side_projects/tensorflow/facial_emotion_recognition/training_data/fer2013.csv"):
    data = pd.read_csv(path)
    try:
        pixels_values = data.pixels.str.split(" ").tolist()
    except:
        pixels_values = data.Pixels.str.split(" ").tolist()
    pixels_values = pd.DataFrame(pixels_values, dtype=int)
    images = pixels_values.values
    images = images - images.mean(axis=1).reshape(-1,1)
    images = np.multiply(images,1/255.0)
    # each_pixel_mean = images.mean(axis=0)
    # each_pixel_std = np.std(images, axis=0)
    # images = np.divide(np.subtract(images,each_pixel_mean), each_pixel_std)

    training = []
    validation = []
    testing = []

    counter = 0
    try:
        emotions = data.emotion.tolist()
        for training_type in data.Usage.tolist():
            if training_type == "Training":
                training.append(np.insert(images[counter],0,emotions[counter]))
            elif training_type == "PrivateTest":
                validation.append(np.insert(images[counter],0,emotions[counter]))
            else:
                testing.append(np.insert(images[counter],0,emotions[counter]))
            counter += 1
    except:
        emotions = data.Emotion.tolist()
        size = np.shape(emotions)
        size = size[0]
        for index in range(size):
            if index < 0.7*size:
                training.append(np.insert(images[index],0,emotions[index]))
            elif index >= 0.7*size and index < 0.85*size:
                testing.append(np.insert(images[index],0,emotions[index]))
            else:
                validation.append(np.insert(images[index],0,emotions[index]))

    return training, validation, testing

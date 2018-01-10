# facial_emotion_recognition
This project is an implementation of a deep neural network in tensorflow and keras for performing facial emotion recognition. It was trained using the publicly available kaggle dataset: Emotion and identity detection from face images (https://www.kaggle.com/c/facial-keypoints-detector).

## Running Facial Emotion Recognition
```
$ python ./run_emotion_recognition.py {optional:-p <path to saved network>} 
```

## Motivation
Humans interact with technology on a daily basis. Technology has become engrained in everyday life. However, there has yet to be an interface implemented for real time feedback between humans and technology. The simplest form of expression that can easily determine user experience is facial expressions. Through training a network for facial emotion recognition, it is possible to create an interface which allows machines to better understand human expression.

## Architecture
The Deep CNN built has the following architecture:
  - Convolutional layer: 20 layers
  - Max pooling layer
  - Convolutional layer: 30 layers
  - Max pooling layer
  - Convolutional layer: 30 layers
  - Convolutional layer: 40 layers
  - Output softmax layer

## Implementation
### Exploratory Data Analysis
In building an efficient machine learning model, it is first required to thoroughly understand the data that will be used for training and testing. Thus, samples were randomly gathered from the data set to visually examine differences in the photos.

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import argparse
import numpy as np

def preprocess_img(img):
    img = img - img.mean()
    # img = np.multiply(img,1/255.0)
    img = img.astype(np.uint32)
    return img
def get_data(path):
    data = pd.read_csv(path)
    try:
        pixels_values = data.pixels.str.split(" ").tolist()
    except:
        pixels_values = data.Pixels.str.split(" ").tolist()
    pixels_values = pd.DataFrame(pixels_values, dtype=int)
    images = pixels_values.values
    images = images - images.mean(axis=1).reshape(-1,1)
    images = np.multiply(images,1/255.0)

    images = images.reshape(-1,48,48)
def read_data(path):
    file = open(path,'r')
    file.readline()

    training_imgs = []
    training_labels = []

    testing_imgs = []
    testing_labels = []

    num_lines = sum([1 for line in file])

    file = open(path,'r')
    file.readline()
    for x in range(int(0.85*num_lines)):
        line = file.readline()
        line = line.strip().split(",")
        line[1] = line[1].strip('"').split()
        img = np.asarray(line[1],dtype=np.uint32)
        img = img - img.mean()
        # img = np.multiply(img,1/255.0)
        img = img.reshape(48,48,1)

        # img = preprocess_img(img)

        training_imgs.append(img)
        label = np.zeros(7)
        label[int(line[0])] = 1
        training_labels.append(label)

    for x in range(int(0.15*num_lines)):
        line = file.readline()
        line = line.strip().split(",")
        line[1] = line[1].strip('"').split()
        img = np.asarray(line[1],dtype=np.uint32)
        img = img - img.mean()
        # img = np.multiply(img,1/255.0)
        img = img.reshape(48,48,1)
        testing_imgs.append(img)
        label = np.zeros(7)
        label[int(line[0])] = 1
        testing_labels.append(label)

    training_imgs = np.asarray(training_imgs)
    training_labels = np.asarray(training_labels)

    testing_imgs = np.asarray(testing_imgs)
    testing_labels = np.asarray(testing_labels)

    return training_imgs,training_labels,testing_imgs,testing_labels

def build_network():
    model = Sequential()
    model.add(Conv2D(20,3,3,activation="relu",input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30,3,3,activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30,3,3,activation="relu"))
    model.add(Conv2D(40,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(40,activation="relu"))
    model.add(Dense(7,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training CNN for facial emotion recognition.')
    parser.add_argument("-p","--training_path",type=str, default="training_data/train.csv" )
    args = parser.parse_args()

    training_imgs,training_labels,testing_imgs,testing_labels = read_data(args.training_path)

    nn = build_network()

    nn.fit(training_imgs,training_labels,batch_size=10,verbose=1,epochs=50,shuffle=True,validation_split=0.20)

    loss, acc = nn.evaluate(testing_imgs,testing_labels)
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    nn.save("keras_models/facial_cnn1.h5")

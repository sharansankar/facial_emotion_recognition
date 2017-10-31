import argparse
import tensorflow as tf
import scipy.misc
import numpy as np

def get_data(path):
    file = open(path,'r')
    file.readline()
    training = []
    testing = []
    validation = []
    for line in file:
        line = line.strip()
        line = line.split(",")
        line[1] = line[1].split()
        if line[-1] == "Training":
            training.append([line[0]] + line[1])
        elif line[-1] == "PrivateTest":
            validation.append([line[0]] + line[1])
        else:
            testing.append([line[0]] + line[1])

    training = np.asarray(training)
    testing = np.asarray(testing)
    validation = np.asarray(validation)

    return training, validation, testing
def get_batch(indata,batch_size):
    """gets batch of data from inputted data. Appends zeros to each sequence to create uniform size
    inputs:
        indata - np.arr: array of lists representing each sequence and class
        batch_size - int: how many training sequences in the batch
    outputs:
        x_batch - list: list of lists representing each training sequence
        y_batch - list: list of x_batch corresponding classes
        sequence_len - list: list of sequence lengths of each sequence in batch
    """
    np.random.shuffle(indata)
    y_batch = []
    x_batch = []
    for x in range(batch_size):
        tmp = [0 for val in range(7)]
        tmp[int(indata[x][0])] = 1
        y_batch.append(tmp)
        x_batch.append(indata[x][1:])
    return x_batch,y_batch

def cnn(x):

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 48, 48, 1])

    #first convolutional layer: maps image to 5 filters
    with tf.name_scope('conv1'):
        W_conv1 = tf.get_variable("W_conv1", shape=[5,5,1,5],initializer=tf.contrib.layers.xavier_initializer())
        # W_conv1 = weight_variable([5,5,1,5])
        b_conv1 = bias_variable([5])
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

    #pooling layer - downsamples by 2x
    with tf.name_scope("pool1"):
        h_pool1 = max_pool_2x2(h_conv1)

    #second convolutional layer: maps 5 filters to another 5 filters
    with tf.name_scope('conv2'):
        W_conv2 = tf.get_variable("W_conv2", shape=[5,5,5,5],initializer=tf.contrib.layers.xavier_initializer())
        # W_conv2 = weight_variable([5,5,5,5])
        b_conv2 = bias_variable([5])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)

    #pooling layer - downsamples by 2x
    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    #third convolutional layer: maps 5 filters to another 5 filters
    with tf.name_scope("conv3"):
        W_conv3 = tf.get_variable("W_conv3", shape=[5,5,5,5],initializer=tf.contrib.layers.xavier_initializer())
        # W_conv3 = weight_variable([5,5,5,5])
        b_conv3 = bias_variable([5])
        h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)

    #pooling layer - downsamples by 2x
    with tf.name_scope("pool3"):
        h_pool3 = max_pool_2x2(h_conv3)

    #final convolutional layer: maps 5 filters to 5 filters
    with tf.name_scope("conv4"):
        W_conv4 = tf.get_variable("W_conv4", shape=[5,5,5,5],initializer=tf.contrib.layers.xavier_initializer())
        # W_conv4 = weight_variable([5,5,5,5])
        b_conv4 = bias_variable([5])
        h_conv4 = tf.nn.relu(conv2d(h_pool3,W_conv4) + b_conv4)

    #fully connected layer: input image has been downsampled 3x, resulting in 6x6 image
    #6x6x5 convolution output mapped to 40 features
    with tf.name_scope("fc1"):
        W_fc1 = tf.get_variable("W_fc1", shape=[6*6*5,40],initializer=tf.contrib.layers.xavier_initializer())
        # W_fc1 = weight_variable([6*6*5,40])
        b_fc1 = bias_variable([40])

        h_conv4_flat = tf.reshape(h_conv4,[-1,6*6*5])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat,W_fc1) + b_fc1)

    #Dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #output layer: map 40 features to the 7 emotion classes
    with tf.name_scope("fc2"):
        W_fc2 = tf.get_variable("W_fc2", shape=[40,7],initializer=tf.contrib.layers.xavier_initializer())
        # W_fc2 = weight_variable([40,7])
        b_fc2 = bias_variable([7])

        y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
    return y_conv, keep_prob

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def train_cnn(train,test,validation,epochs=10000,batch_size=55):
    x = tf.placeholder(tf.float32,[None,48*48])
    y_ = tf.placeholder(tf.float32,[None,7])

    #build graph
    y_conv, keep_prob = cnn(x)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope("adam_optimizer"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
        correct_prediction = tf.cast(correct_prediction,tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            batch_x, batch_y = get_batch(train,batch_size)
            train_step.run(feed_dict={x:batch_x, y_:batch_y, keep_prob:0.6})
            if i % 100 == 0:
                val_x, val_y = get_batch(validation,batch_size)
                train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob:0.0})
                val_accuracy = accuracy.eval(feed_dict={x:val_x, y_:val_y, keep_prob:0.0})
                print('step %d, training accuracy %g, validation accuracy %g' % (i, train_accuracy, val_accuracy))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training CNN for facial emotion recognition.')
    parser.add_argument("-p","--training_path",type=str )
    args = parser.parse_args()

    train,valid,test = get_data(args.training_path)
    # test_x, test_y = get_batch(test,5)
    # print(test)
    # print(test_y)
    # print(test_x)
    train_cnn(train,test,valid)

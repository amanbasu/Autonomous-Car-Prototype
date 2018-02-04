import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn 
from tensorflow.python.ops import rnn, rnn_cell

dropout = 0.8
n_classes = 3
batch_size = 64
hm_epochs = 30

chunk_size = 32
n_chunks = 14
rnn_size = 64

x = tf.placeholder('float',[None,14,32,1])
y = tf.placeholder('float')

def get_data():
    df = pd.read_csv('indoor_track_data.csv')
    df.drop('Unnamed: 0',axis=1,inplace=True)

    df = df[df['448']!='S']
    df = df[df['448']!='s']

    X = df.drop(str(14*32), axis=1)
    y = df[str(14*32)]

    X_data = X.values.reshape(-1,448)
    X_data = X_data.reshape(X_data.shape[0], 14, 32, 1)
    X_data = X_data.astype('float32')
    X_data /= 255

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    y_data = tf.contrib.keras.utils.to_categorical(y, 3)

    X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, random_state=0, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test,y_test, random_state=0, test_size=0.5)
    
    return [X_train, X_test, X_val, y_train, y_test, y_val]

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def cnn_model(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([512,128])),
               'out':tf.Variable(tf.random_normal([128, n_classes]))}
    
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([128])),
              'out':tf.Variable(tf.random_normal([n_classes]))}
        
    x = tf.reshape(x, shape=[-1,14,32,1])
        
    conv1 = conv2d(x, weights['W_conv1']) + biases['b_conv1']
    conv1 = tf.nn.relu(conv1)
    conv1 = maxpool2d(conv1)
    conv1 = tf.nn.dropout(conv1, dropout)
    print("Conv1 shape:", conv1.shape)

    conv2 = conv2d(conv1, weights['W_conv2']) + biases['b_conv2']
    conv2 = tf.nn.relu(conv2)
    conv2 = maxpool2d(conv2)
    conv2 = tf.nn.dropout(conv2, dropout)
    print("Conv2 shape:", conv2.shape)

    rnn_input = tf.transpose(conv2,[1,0,2,3])
    rnn_input = tf.reshape(rnn_input, [-1, 4*64])
    rnn_input = tf.split(rnn_input, 4, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    rnn_output, final_state = rnn.static_rnn(lstm_cell, rnn_input, dtype=tf.float32)
    # print("rnn_output shape:", rnn_output.shape)

    fc = tf.reshape(rnn_output, [-1, 512])
    fc = tf.matmul(fc, weights['W_fc']) + biases['b_fc']
    fc = tf.nn.relu(fc)
    fc = tf.nn.dropout(fc, dropout)
    print("fc shape:",fc.shape)
    
    output = tf.matmul(fc, weights['out']) + biases['out']
    print("output shape:",output.shape)
    return output

def train_model(x):
    prediction = cnn_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    [X_train, X_test, X_val, y_train, y_test, y_val] = get_data()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epochs in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(X_train.shape[0]/batch_size)):
                batch_x = X_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train[i*batch_size:(i+1)*batch_size]
                
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c

            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Epoch:', epochs+1, '/', hm_epochs, ' Loss:', epoch_loss, ' Accuracy:', accuracy.eval({x: X_val, y: y_val}))
            
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ', accuracy.eval({x: X_test, y: y_test}))
train_model(x)

import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn 
from tensorflow.python.ops import rnn, rnn_cell

hm_epochs = 10
n_classes = 4
batch_size = 64

chunk_size = 32
n_chunks = 14
rnn_size = 128

x = tf.placeholder('float',[None, n_chunks, chunk_size])
y = tf.placeholder('float')

def get_data():
    df = pd.read_csv('data.csv')
    df.drop('Unnamed: 0',axis=1,inplace=True)

    df = df[df['448']!='S']

    X = df.drop(str(14*32), axis=1)
    y = df[str(14*32)]

    X_data = X.values.reshape(-1,448)
    X_data = X_data.reshape(X_data.shape[0], 14, 32, 1)
    X_data = X_data.astype('float32')
    X_data /= 255

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    y_data = tf.contrib.keras.utils.to_categorical(y, 4)

    X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, random_state=0, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test,y_test, random_state=0, test_size=0.5)
    
    return [X_train, X_test, X_val, y_train, y_test, y_val]

def rnn_model(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    # x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)
    
    # single RNN layer
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    rnn_outputs, final_state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
   
    # multiple RNN layers 
    '''
    num_layers = 3

    stacked_rnn = []
    for _ in range(num_layers):
        stacked_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    rnn_outputs, final_state = tf.nn.static_rnn(cell, x, dtype=tf.float32)
	'''

    output = tf.add(tf.matmul(rnn_outputs[-1], layer['weights']), layer['biases'])
    
    return output

def train_model(x):
    prediction = rnn_model(x)

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
                
                batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c

            # correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            # accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            # print('Epoch:', epochs+1, '/', hm_epochs, ' Loss:', epoch_loss, ' Accuracy:', accuracy.eval({x: X_val.reshape((-1, n_chunks, chunk_size)), y: y_val}))   
            print('Epoch:', epochs+1, '/', hm_epochs, ' Loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ', accuracy.eval({x: X_test.reshape((-1, n_chunks, chunk_size)), y: y_test}))
train_model(x)
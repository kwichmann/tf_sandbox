# Model is the same as in mnist.py, but tweaked to fit

import tensorflow as tf
import numpy as np

# But use data from pickle (it is assumed you have run make_sentiment_data.py locally)

import pickle

with open("sentiment_set.pickle", "rb") as f:
    train_x, train_y, test_x, test_y = pickle.load(f)

nodes1 = 100
nodes2 = 100
nodes3 = 100

classes = 2

batch_size = 100

x = tf.placeholder("float", [None, 423])
y = tf.placeholder("float")

def model(data):
    layer1 = {"weights": tf.Variable(tf.random_normal([423, nodes1])),
              "biases": tf.Variable(tf.random_normal([nodes1]))}
    layer2 = {"weights": tf.Variable(tf.random_normal([nodes1, nodes2])),
              "biases": tf.Variable(tf.random_normal([nodes2]))}
    layer3 = {"weights": tf.Variable(tf.random_normal([nodes2, nodes3])),
              "biases": tf.Variable(tf.random_normal([nodes3]))}
    output = {"weights": tf.Variable(tf.random_normal([nodes3, classes])),
              "biases": tf.Variable(tf.random_normal([classes]))}
    
    l1 = tf.add(tf.matmul(data, layer1["weights"]), layer1["biases"])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, layer2["weights"]), layer2["biases"])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, layer3["weights"]), layer3["biases"])
    l3 = tf.nn.relu(l3)
    
    return tf.add(tf.matmul(l3, output["weights"]), output["biases"])

def train(x):
    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # Learning rate defaults to 0.001
    
    epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            
            print("Epoch", epoch, "out of", epochs, "completed. Loss", epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))       
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))            
            
train(x)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

nodes1 = 100
nodes2 = 100
nodes3 = 100

classes = 10

batch_size = 100

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")

def model(data):
    layer1 = {"weights": tf.Variable(tf.random_normal([784, nodes1])),
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
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            
            print("Epoch", epoch, "out of", epochs, "completed. Loss", epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))       
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))            
            
train(x)
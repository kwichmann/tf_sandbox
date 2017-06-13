import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

epochs = 10
classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28

rnn_size = 128

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")

def recurrent_model(data):
    layer = {"weights": tf.Variable(tf.random_normal([rnn_size, classes])),
             "biases": tf.Variable(tf.random_normal([classes]))}
    
    return tf.add(tf.matmul(???, layer["weights"]), layer["biases"])

def train(x):
    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # Learning rate defaults to 0.001
    
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

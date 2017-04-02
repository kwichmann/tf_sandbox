import tensorflow as tf

x1 = tf.constant(3)
x2 = tf.constant(5)

result = tf.multiply(x1, x2)

with tf.Session() as sess:
    print(sess.run(result))

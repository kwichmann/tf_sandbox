import tensorflow as tf

a = tf.constant(value=3.0, dtype=tf.float32, name="a")
b = tf.constant(value=4.0, dtype=tf.float32, name="b")
c = tf.sqrt(tf.add(tf.square(a), tf.square(b)))

with tf.Session() as sess:
    print(*sess.run([a, b, c]), sep="\n")

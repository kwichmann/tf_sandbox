import tensorflow as tf

W = tf.Variable(initial_value=[.3], dtype=tf.float32, name='W')
b = tf.Variable(initial_value=[-.3], dtype=tf.float32, name='b')
x = tf.placeholder(tf.float32, name='x')
linear_model = W * x + b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(linear_model, feed_dict={x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
    
dW,db = tf.gradients(loss, [W,b])
print(dW, db, sep='\n')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(*sess.run([dW,db], feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]}), sep="\n")

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("initial params", *sess.run([W, b]), sep='\n')
    for i in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

    print("optimized params after 1000 updates", *sess.run([W, b]), sep='\n')
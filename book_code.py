import tensorflow as tf
import numpy as np

# # # Variables Section
# # var1 = tf.Variable(initial_value=10, name="mohamed")
# # var2 = tf.random_normal((2,3), mean=2)
# # print("pre run {}".format(var2))

# # # initiliaze variables 
# # init = tf.global_variables_initializer()

# # with tf.Session() as sess:
# #     sess.run(init)
# #     post_var = sess.run(var2)

# # print("post run {}".format(post_var))

# print("########################")

# # Placeholder Section

# x_data = np.random.randn(5,10)
# w_data = np.random.randn(10,1)

# with tf.Graph().as_default():
#     x = tf.placeholder(dtype=tf.int32 ,shape=(5,10))
#     w = tf.placeholder(dtype=tf.int32, shape=(10,1))
#     b = tf.fill((5,1),1)

#     xw = tf.matmul(x,w)

#     xwb = xw + b
#     s = tf.reduce_max(xwb)
#     with tf.Session() as sess:
#         outs = sess.run(s, feed_dict={x : x_data , w : w_data})

# print("outs {}".format(outs))


# # linear regression
# x = tf.placeholder(tf.float32,shape=[None,3])
# y_true = tf.placeholder(tf.float32,shape=None)
# w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
# b = tf.Variable(0,dtype=tf.float32,name='bias')
# y_pred = tf.matmul(w,tf.transpose(x)) + b

# # what we use to determine our loss
# loss = tf.reduce_mean(tf.square(y_true - y_pred))

# # Cross entropy
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits= y_pred)

"""
Optimizers - Gradient Descent Optimizer
update the set of weights iteratively in a way that decreases the 
loss over time.
"""

"""
A more popular technique is the stochastic gradient descent (SGD), where instead of
feeding the entire dataset to the algorithm for the computation of each step, a subset
of the data is sampled sequentially. The number of samples ranges from one sample at
a time to a few hundred, but the most common sizes are between around 50 to
around 500 (usually referred to as mini-batches)
"""

# Example 1 Linear Regression

NUM_STEPS = 10

# Create data
"""For this exercise we will generate synthetic data using NumPy. We create 2,000 samples
of x, a vector with three features, take the inner product of each x sample with a
set of weights w ([0.3, 0.5, 0.1]), and add a bias term b (–0.2) and Gaussian noise to
the result"""

x_data = np.random.randn(2000,3)
w_real = np.random.randn(3, 1)
b_real = -0.2

noise = np.random.randn(1 , 2000) * 0.1
y_data = np.dot(w_real,x_data)+ b_real + noise
print(y_data)

g = tf.Graph()
wb = []

with g.as_default():
    # define placeholders
    x = tf.placeholder(tf.float32, shape= [None,3])
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope('inference') as scope:
        """In this example we initialize both the weights
           and the bias with zeros"""
        w = tf.Variable([[0,0,0]],dtype=tf.float32, name='weight')
        b = tf.Variable(0 , dtype=tf.float32 , name='bias')
        y_pred = tf.matmul(w, tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)
    
    # now we initialize variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train ,feed_dict={x: x_data , y_true : y_data})
            if step % 5 == 0:
                print(step, sess.run([w,b]))
                wb.append(sess.run([w,b]))
        print(10 , sess.run([w,b]))



# Logistic Regression - 0 or 1 
# used sigmoid function
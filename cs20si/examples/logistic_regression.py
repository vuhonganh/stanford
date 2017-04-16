import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets("/tmp/mnist", one_hot=True)

# NOTE: each image in MNIST has resolution 28*28 = 784 and each turns to
# a vector of 784 dims
vec_dim = 784
num_classes = 10
batch_size = 128
num_epochs = 24
learning_rate = 0.01

# each features X has shape (batch_size, vec_dim)
# as we pass in one batch at a time
# note that in tensorflow shape is passed by a list of int
X = tf.placeholder(tf.float32, [batch_size, vec_dim])

# each labels Y has shape (batch_size, num_classes)
# as we pass in one batch at a time
Y = tf.placeholder(tf.float32, [batch_size, num_classes])

w = tf.Variable(tf.random_normal(shape=[vec_dim, num_classes], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, num_classes]), name='bias')

logits = tf.matmul(X, w) + b

# loss of the batch
loss_batch = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

# loss is computed by taking average
loss = tf.reduce_mean(loss_batch, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    num_batches = int(MNIST.train.num_examples / batch_size)
    print("Training...")
    for i in range(num_epochs):
        cur_loss = 0.0  # current loss for this epoch
        for j in range(num_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)  # type of X_batch and Y_batch are numpy.ndarray
            _, l = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            cur_loss += l
        cur_loss /= num_batches
        print("Epoch {0}: loss = {1}".format(i, cur_loss))

    num_batches_test = int(MNIST.test.num_examples / batch_size)
    acc = 0.0
    print("Testing...")
    for i in range(num_batches_test):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        logits_batch = sess.run(logits, feed_dict={X: X_batch, Y: Y_batch})
        scores = tf.nn.softmax(logits_batch)
        pred_labels = tf.argmax(scores, axis=1)  # note that scores has shape (batch_size, num_classes)
        correct_labels = tf.argmax(Y_batch, axis=1)  # one hot -> idx of value 1 is also max (the rest = 0)
        correct_preds = tf.equal(pred_labels, correct_labels)
        acc += tf.reduce_sum(tf.cast(correct_preds, tf.float32)).eval()
    print("total accuracy on test set = {0}".format(acc / MNIST.test.num_examples))

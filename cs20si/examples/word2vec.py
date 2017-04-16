import tensorflow as tf
from process_data import process_data

BATCH_SIZE = 128
VOCAB_SIZE = 50000
EMBED_SIZE = 128
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss
NUM_SAMPLED = 64
SKIP_WINDOW = 1

def word2vec(batch_gen):
    """define word2vec and train it"""
    # Step 1: define the placeholders for input and output
    # center_words have to be int to work on embedding lookup
    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])


    # Step 2: define weights. In word2vec, it's actually the weights that we care about
    # vocab size x embed size
    # initialized to random uniform -1 to 1
    embed_matrix = tf.Variable(tf.random_uniform(shape=[VOCAB_SIZE, EMBED_SIZE], minval=-1.0,  maxval=1.0))
    # TOO DO

    # Step 3: define the inference
    # get the embed of input words using tf.nn.embedding_lookup
    embed = tf.nn.embedding_lookup(params=embed_matrix, ids=center_words, name='embed')
    # embed has shape (BATCH_SIZE, EMBED_SIZE)

    # Step 4: construct variables for NCE loss
    # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
    # nce_weight (vocab size x embed size), intialized to truncated_normal stddev=1.0 / (EMBED_SIZE ** 0.5)
    # bias: vocab size, initialized to 0

    nce_weight = tf.Variable(tf.truncated_normal(shape=[VOCAB_SIZE, EMBED_SIZE], mean=0.0, stddev=1.0/(EMBED_SIZE ** 0.5)))
    nce_bias = tf.Variable(tf.zeros(shape=[VOCAB_SIZE]))
    # define loss function to be NCE loss function
    # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
    # need to get the mean accross the batch

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed,
                                         num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE))

    # Step 5: define optimizer

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # TO DO: initialize variables
        sess.run(init)
        total_loss = 0.0  # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./my_graph/no_frills/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            # TO DO: create feed_dict, run optimizer, fetch loss_batch
            _, loss_batch = sess.run(fetches=[optimizer, loss], feed_dict={center_words : centers, target_words : targets})
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / (index + 1)))

        writer.close()

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()
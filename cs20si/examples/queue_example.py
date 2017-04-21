import numpy as np
import tensorflow as tf

N_SAMPLES = 5
NUM_THREADS = 2

# add some dummy data
all_data = 10 * np.random.randn(N_SAMPLES, 4) + 1
all_target = np.random.randint(0, 2, size=N_SAMPLES)

# create queue: dtypes and shapes specify types and shapes of data and labels respectively
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

# common practice is to enqueue all data at once, but dequeue one by one
enqueue_op = queue.enqueue_many([all_data, all_target])
data_sample, label_sample = queue.dequeue()

# create a number of threads cooperating to enqueue tensors in the same queue
qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)

with tf.Session() as sess:
    # create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    # Run the training loop, controlling termination with the coordinator.
    for step in range(100): # do 100 iterations in training loop...
        if coord.should_stop():
            break
        one_data, one_label = sess.run([data_sample, label_sample])
        # and also run training operations on the one_data and one_label above: sess.run([train_ops])

    # When done, ask the threads to stop.
    coord.request_stop()

    # And wait for them to actually do it.
    coord.join(enqueue_threads)
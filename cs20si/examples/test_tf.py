import tensorflow as tf

a = tf.constant([2, 2], name="a")
b = tf.constant([[0, 1], [2, 3]], name="b")

x = tf.add(a, b, name='add')

y =  tf.multiply(a, b, name="mul")


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    out_x, out_y = sess.run([x, y])
    print(out_x)
    print(out_y)
writer.close()

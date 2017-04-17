import tensorflow as tf
from process_data import process_data
import os
from tensorflow.contrib.tensorboard.plugins import projector

BATCH_SIZE = 128
VOCAB_SIZE = 50000
EMBED_SIZE = 128
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 50000
SKIP_STEP = 2000  # how many steps to skip before reporting the loss
NUM_SAMPLED = 64
SKIP_WINDOW = 1


class SkipGramModel:
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        with tf.name_scope('data'):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    def _create_embedding(self):
        with tf.name_scope('embed'):
            self.embed_matrix = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.embed_size],
                                                              minval=-1.0,  maxval=1.0), name='embed_matrix')

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.embed = tf.nn.embedding_lookup(params=self.embed_matrix, ids=self.center_words, name='embed')
            nce_weight = tf.Variable(tf.truncated_normal(shape=[self.vocab_size, self.embed_size],
                                                         mean=0.0,
                                                         stddev=1.0/(self.embed_size ** 0.5)),
                                     name='nce_weight')
            nce_bias = tf.Variable(tf.zeros(shape=[self.vocab_size]), name='nce_bias')
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias,
                                                      labels=self.target_words, inputs=self.embed,
                                                      num_sampled=NUM_SAMPLED, num_classes=self.vocab_size),
                                       name='loss')

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                                        global_step=self.global_step)

    def _create_summary(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

def train_model(model, batch_gen, num_train_steps):
    saver = tf.train.Saver()  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # switch to average_loss because using saved session here
        average_loss = 0.0

        writer = tf.summary.FileWriter('./my_graph/word2vec_visualized/', sess.graph)

        initial_step = model.global_step.eval()

        for index in range(initial_step, initial_step + num_train_steps):
            centers, targets = next(batch_gen)
            # TO DO: create feed_dict, run optimizer, fetch loss_batch
            _, loss_batch = sess.run(fetches=[model.optimizer, model.loss],
                                     feed_dict={model.center_words : centers, model.target_words : targets})
            average_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index + 1, average_loss / SKIP_STEP))
                average_loss = 0.0  # finish this batch, reset average loss to 0 for next batch
                if not os.path.exists('./checkpoints/'):
                    os.makedirs('./checkpoints/')
                saver.save(sess, 'checkpoints/skip-gram', index)

        # code to visualize the embeddings.
        final_embed_matrix = sess.run(model.embed_matrix)

        # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = 'processed/vocab_1000.tsv'

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'processed/model3.ckpt', 1)

        writer.close()


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    train_model(model, batch_gen, NUM_TRAIN_STEPS)

if __name__ == '__main__':
    main()
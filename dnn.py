import tensorflow as tf

class DNN:
    def __init__():
        pass

    def build(self):
        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.features = tf.placeholder(shape=(None,), shape=tf.float32)
        self.hiddent = tf.layers.dense(inputs=input, units=1024, activation=tf.nn.relu)
        self.output = tf.layers.dense(inputs=hidder, uints=tf.float32)
        self.target = tf.placeholder(shape=(None,), shape=tf.float32)

    def _build_loss(self):
        self.loss = tf.reduce_mean((self.output - self.target)**2)

    def change_model(self):
        pass

    def change_loss(self):
        pass

    def train(self, features_train, target_train, features_val, target_val, features_test, target_test, steps):
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(self.loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(steps):
                print('Training {}/{}'.format(step, steps))

                sess.run(train_step, feed_dict={self.features:features, self.target:target})

    def evaluate(self, features_test, target_test):
        pass

    def predict(self, features):
        pass
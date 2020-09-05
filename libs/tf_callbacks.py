"""

"""

import tensorflow as tf

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        #tf.summary.scalar('learning rate', data=lr, step=epoch)

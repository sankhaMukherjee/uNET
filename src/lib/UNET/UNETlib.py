from tqdm     import tqdm
from logs     import logDecorator as lD
from datetime import datetime     as dt

import json, os
import numpy      as np
import tensorflow as tf


config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.UNETlib.UNET'

class UNET():

    @lD.log(logBase + '.__init__')
    def __init__(logger, self, inpShape, outShape, layers, activations):
        
        self.restorePoints = []
        self.logPoints     = []

        self.Inp  = tf.placeholder(dtype=tf.float32, shape=inpShape )
        self.Out  = tf.placeholder(dtype=tf.float32, shape=outShape )

        # ------------------------------------------------------------
        # Generate a dense network
        # ------------------------------------------------------------
        self.nn = self.Inp*1
        for i, (l, a) in enumerate(zip(layers, activations)):
            self.nn = tf.layers.dense(self.nn, l, activation=a, name='dense_{:05d}'.format(i))

        # ------------------------------------------------------------
        # Calculate MSE
        # ------------------------------------------------------------        
        self.Err = tf.reduce_mean((self.nn - self.Out)**2, name='error')

        # ------------------------------------------------------------
        # Generate other misc operations
        # ------------------------------------------------------------
        self.Opt  = tf.train.AdamOptimizer().minimize( self.Err )
        self.init = tf.global_variables_initializer()
        
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        return

    @lD.log(logBase + '.saveModel')
    def saveModel(logger, self, sess):
        now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        modelFolder = '../models/{}'.format(now)
        os.makedirs(modelFolder)

        path = self.saver.save( sess, os.path.join( modelFolder, 'model.ckpt' ) )
        self.restorePoints.append(path)

        return path

    @lD.log(logBase + '.restoreModel')
    def restoreModel(logger, self, sess, restorePoint):
        if restorePoint is not None:
            try:
                self.saver.restore(sess, restorePoint)
            except Exception as e:
                logger.error('Unable to restore the session at [{}]:{}'.format(
                    restorePoint, str(e)))
                return False
        else:
            return False

        return True

    @lD.log(logBase + '.fit')
    def fit(logger, self, X, y, Niter=101, restorePoint=None):

        try:
            with tf.Session() as sess:
                sess.run(self.init)

                # Try to restore an older checkpoint
                # ---------------------------------------
                self.restoreModel(sess, restorePoint)
                
                for i in tqdm(range(Niter)):
                    _, Err = sess.run(
                            [self.Opt, self.Err], 
                            feed_dict = {
                                self.Inp    : X,
                                self.Out    : y})

                self.saveModel(sess)

        except Exception as e:
            logger.error('Unable to fit the model: {}'.format( str(e) ))

        return

    @lD.log(logBase + '.predict')
    def predict(logger, self, X, restorePoint=None):
        try:
            with tf.Session() as sess:
                sess.run(self.init)

                # Try to restore an older checkpoint
                # ---------------------------------------
                restored = self.restoreModel(sess, restorePoint)
                if not restored:
                    logger.warning('Predicting without restoring a previous setting')

                yHat = sess.run(self.nn, 
                        feed_dict = { self.Inp    : X})

                return yHat
                
        except Exception as e:
            logger.error('Unable to make a prediction: {}'.format( str(e) ))

        return

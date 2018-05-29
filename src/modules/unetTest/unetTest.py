from logs import logDecorator as lD 
import json

from lib.UNET import UNETlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.unetTest.unetTest'


@lD.log(logBase + '.testUNet')
def testUNet(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    X = np.random.random((1000, 2))
    y = X[:, 0]*2 - X[:, 1]*3
    y = y.reshape((-1, 1))

    inpShape     = (None, 2)
    outShape     = (None, 1)
    layers       = [3, 5, 1]
    activations  = [tf.tanh, tf.tanh, None]

    print('Generating the U-net')
    unet = UNETlib.UNET(inpShape, outShape, layers, activations)
    
    print('Fitting the model')
    unet.fit(X, y, 5000)

    print('Making a prediction')
    yHat = unet.predict(X, restorePoint=unet.restorePoints[-1])

    print('Plotting the result')
    plt.plot(y, yHat, '.')
    plt.savefig('../results/testImg.png')
    plt.close('all')


    print('Finished with all the functions')


    return

@lD.log(logBase + '.main')
def main(logger):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    testUNet()

    return


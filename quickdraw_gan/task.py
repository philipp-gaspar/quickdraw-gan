import os
import argparse
import tensorflow as tf
import numpy as np

import utils
from model import WGAN_GP

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_args():
    """Argument parser.

    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='number of times to go through the data, default=100')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--grad-weight',
        type=float,
        default=10.,
        help='regularization constant of gradient penalty')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args, _ = parser.parse_known_args()
    return args

def train_gan(args):
    """Train an WGAN.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in utils.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
        args: dictionr of arguments - see get_args() for details
    """
    data = utils.load_dataset()
    data = data[0:100,:,:]

    # setup tensorboard callback
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'), histogram_freq=1)

    # train model
    model = WGAN_GP()
    generator, losses = model.train(data, args.num_epochs,
        args.batch_size, args.grad_weight)

    export_path = os.path.join(args.job_dir, 'keras_export')
    # tf.keras.experimental.export_saved_model(model, export_path)
    print('Model exported to: %s' % export_path)



if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_gan(args)

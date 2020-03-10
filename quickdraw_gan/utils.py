import os
import sys
import numpy as np
import tensorflow as tf

DATA_DIR = os.path.join(os.environ['HOME'], 'DataScience', 'quickdraw-gan',
    'data', 'raw')
TRAINING_FILE = 'cat.npy'

def check_file(input_file):
    """Check if file exists.

    Args:
        input_file: complete file path
    """
    try:
        assert os.path.isfile(input_file)
    except AssertionError:
        print('FILE NOT FOUND!')
        print('%s' % input_file)
        sys.exit()

def load_dataset():
    """Loads numpy bitmap dataset.

    Returns:
        data: 3D array of reshaped data (entries, rows, cols)
    """
    input_file = os.path.join(DATA_DIR, TRAINING_FILE)
    check_file(input_file)

    data = np.load(input_file)
    data = np.reshape(data, newshape=(data.shape[0], 28, 28))

    return data

def prepare_dataset(data, batch_size):
    """Convert NumPy array into a TensorFlow Dataset for training.

    Args:
        data: input data array (entries, rows, cols, channels)
    """
    NUM_SAMPLES = data.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices(data)

    # batch and shuffle data
    tf.random.set_seed(13)
    train_dataset = train_dataset.shuffle(NUM_SAMPLES).batch(batch_size,
        drop_remainder=True)

    return train_dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import logging
import math


import numpy as np

import tensorflow as tf
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger

_logger = create_logger("cleverhans.utils.tf")
_logger.setLevel(logging.INFO)


def model_eval_targetacc(sess, x, y, y_target, predictions, X_test=None, Y_test_target=None,
               feed=None, args=None):
  """
  Compute the accuracy of a TF model on some data
  :param sess: TF session to use
  :param x: input placeholder
  :param y: output placeholder (for labels)
  :param predictions: model output predictions
  :param X_test: numpy array with training inputs
  :param Y_test: numpy array with training outputs
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `batch_size`
  :return: a float with the accuracy value
  """
  args = _ArgsWrapper(args or {})

  assert args.batch_size, "Batch size was not given in args dict"
  if X_test is None or Y_test_target is None:
    raise ValueError("X_test argument and Y_test argument "
                     "must be supplied.")

  # Define accuracy symbolically
  if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
    correct_preds = tf.equal(tf.argmax(y, axis=-1),
                             tf.argmax(predictions, axis=-1))
  else:
    correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1),
                             tf.argmax(predictions,
                                       axis=tf.rank(predictions) - 1))

  # Init result var
  accuracy = 0.0

  with sess.as_default():
    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)

    X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    Y_cur = np.zeros((args.batch_size,) + Y_test_target.shape[1:],
                     dtype=Y_test_target.dtype)
    for batch in range(nb_batches):
      if batch % 100 == 0 and batch > 0:
        _logger.debug("Batch " + str(batch))

      # Must not use the `batch_indices` function here, because it
      # repeats some examples.
      # It's acceptable to repeat during training, but not eval.
      start = batch * args.batch_size
      end = min(len(X_test), start + args.batch_size)

      # The last batch may be smaller than all others. This should not
      # affect the accuarcy disproportionately.
      cur_batch_size = end - start
      X_cur[:cur_batch_size] = X_test[start:end]
      Y_cur[:cur_batch_size] = Y_test_target[start:end]
      feed_dict = {x: X_cur, y: Y_cur, y_target: Y_cur}
      if feed is not None:
        feed_dict.update(feed)
      cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)

      accuracy += cur_corr_preds[:cur_batch_size].sum()

    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)

  return accuracy
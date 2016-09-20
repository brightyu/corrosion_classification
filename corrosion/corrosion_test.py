from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import corrosion

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/djatha/honors_thesis/corrosion_smaller_dataset/corrosion_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/djatha/honors_thesis/corrosion_smaller_dataset/corrosion_train',
                           """Directory where to read model checkpoints.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/corrosion_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    #coord = tf.train.Coordinator()
    #try:
    #  threads = []
    #  for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
    #                                     start=True))

    #  num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    #  true_count = 0  # Counts the number of correct predictions.
    #  total_sample_count = num_iter * FLAGS.batch_size
    #  step = 0
    #  while step < num_iter and not coord.should_stop():
    predictions = sess.run([top_k_op])
    print(predictions)
    #true_count += np.sum(predictions)
    #step += 1

    # Compute precision @ 1.
    #precision = true_count / total_sample_count
    #print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

    #summary = tf.Summary()
    #summary.ParseFromString(sess.run(summary_op))
    #summary.value.add(tag='Precision @ 1', simple_value=precision)
    #summary_writer.add_summary(summary, global_step)
    #except Exception as e:  # pylint: disable=broad-except
    #  coord.request_stop(e)

    #coord.request_stop()
    #coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval corrosion for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    #eval_data = FLAGS.eval_data == 'test'
    #images, labels = corrosion.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = corrosion.inference(images)

    # Calculate predictions.
    #top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_op = tf.nn.top_k(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        corrosion.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    eval_once(saver, summary_writer, top_k_op, summary_op)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()

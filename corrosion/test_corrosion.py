import numpy as np
import cv2
import os

import tensorflow as tf
import corrosion

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/djatha/honors_thesis/tmp/corrosion_test_eval',
                           """Directory where to write event logs.""")
#tf.app.flags.DEFINE_string('eval_data', 'test',
#                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/djatha/honors_thesis/tmp/corrosion_train',
                           """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
#                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 38482,
                            """Number of examples to run.""")
#tf.app.flags.DEFINE_boolean('run_once', False,
#                         """Whether to run eval only once.""")

def restore_vars(saver, sess, chkpt_dir):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.initialize_all_variables())

    checkpoint_dir = chkpt_dir

    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass

    path = tf.train.get_checkpoint_state(checkpoint_dir)
    #print("path1 = ",path)
    #path = tf.train.latest_checkpoint(checkpoint_dir)
    print(checkpoint_dir,"path = ",path)
    if path is None:
        return False
    else:
        saver.restore(sess, path.model_checkpoint_path)
        return True


imgs_place = tf.placeholder(tf.float32, shape = [64,64,3])
images = tf.reshape(imgs_place, (1,64,64,3))

#saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
logits = corrosion.inference(images)

saver = tf.train.Saver()
#restored = restore_vars(saver, sess, FLAGS.checkpoint_dir)
with tf.Session() as sess:
    #saver = tf.train.Saver()
    restored = restore_vars(saver, sess, FLAGS.checkpoint_dir)
    #img = tf.image.decode_jpeg('/home/djatha/Pictures/corrosion_test/1.jpg')
    img = cv2.imread('/home/djatha/Pictures/corrosion_test/6.jpg')
    cv2.imshow('test', img)
    img = img.astype(np.float32)
    img = img / 255.0

    # more checks here
    min_ind_y = 0
    max_ind_y = 64
    while max_ind_y < img.shape[0]:
        img_y = img[min_ind_y:max_ind_y]
        min_ind_x = 0
        max_ind_x = 64
        while max_ind_x < img.shape[1]:
            img_xy = img_y[:, min_ind_x:max_ind_x]
            #cv2.imshow('first', img_xy)
            #cv2.waitKey()
            logit_val = sess.run(logits, feed_dict={imgs_place: img_xy})
            print(logit_val)
            if logit_val[0][0] > logit_val[0][1]:
                cv2.rectangle(img, (min_ind_x+3, min_ind_y+3), (max_ind_x-3, max_ind_y-3), (0,255,0))
            else:
                cv2.rectangle(img, (min_ind_x+3, min_ind_y+3), (max_ind_x-3, max_ind_y-3), (0,0,255), thickness=2)
            min_ind_x = max_ind_x
            max_ind_x += 64
        min_ind_y = max_ind_y
        max_ind_y += 64
    cv2.imshow('result', img)
    cv2.waitKey()


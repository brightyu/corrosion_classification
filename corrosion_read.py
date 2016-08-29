import glob
import os
import numpy as np
import skimage.io as io
from tensorflow.python.framework import dtypes

class DataSet(object):
    def __init__(self, images, labels, reshape=True, dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num_examples, rows, columns, channels]
        # to [num_examples, rows*columns*channels] 
        if reshape:
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2] * images.shape[3])

        if dtype == dtypes.float32:
            # Convert from [0, 255] to [0.0, 1.0]
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        return

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

        
def read_data(path, mode='train'):
    labels_dict = read_labels(path, mode)
    path = os.path.join(path, 'images')
    images = glob.glob(os.path.join(path, '*.jpg'))
    num_images = len(images)
    imgs = np.zeros([num_images,64,64,3])
    labels = np.zeros([num_images, 2], dtype=np.float32)
    for ind, image in enumerate(images):
        imgs[ind] = io.imread(image)
        image_key = image.strip(path).strip('.jpg')
        labels[ind] = labels_dict[image_key]

    return DataSet(imgs, labels)



def read_labels(path, mode='train'):
    label_file = open(os.path.join(path, mode + '.txt'))
    label_dict = {}
    for line in label_file:
        line = line.split(' ')
        label_arr = np.zeros([2,], dtype=np.float32)
        ind = int(line[1].strip('\n'))
        label_dict[line[0]] = label_arr[ind] = 1.0
    return label_dict


import numpy as np
import tensorflow as tf
import threading

from utils.data_utils import im_path_to_arr
from utils.data_utils import normalize_image
from utils.data_utils import preprocess_labels


def get_batch_train(ex_path, batch_size, num_batches, patch_size):
    data = [None] * 4
    half_patch = patch_size // 2

    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0].split('_')[-1]
        image = im_path_to_arr(im_path)
        if im_type == 't1':
            data[0] = normalize_image(image)
        if im_type == 't1c' or im_type == 't1ce':
            data[1] = normalize_image(image)
        if im_type == 't2':
            data[2] = normalize_image(image)
        if im_type == 'flair' or im_type == 'fla':
            data[3] = normalize_image(image)
        if im_type == 'tumor' or im_type == 'seg':
            labels = preprocess_labels(image)

    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    trimmed_data = data[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch, :]
    trimmed_labels = labels[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch]

    bg = np.where((trimmed_labels == 0) & (trimmed_data[..., 0] != 0))
    fg = np.where((trimmed_labels > 0) & (trimmed_data[..., 0] != 0))

    num_bg = len(bg[0])
    num_fg = len(fg[0])

    for _ in range(num_batches):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):
            if np.random.rand() <= 0.5:
                idx = np.random.randint(num_bg)
                i = bg[0][idx]
                j = bg[1][idx]
                k = bg[2][idx]
                x = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
                y = labels[i:i + patch_size, j:j + patch_size, k:k + patch_size]
            else:
                idx = np.random.randint(num_fg)
                i = fg[0][idx]
                j = fg[1][idx]
                k = fg[2][idx]
                x = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
                y = labels[i:i + patch_size, j:j + patch_size, k:k + patch_size]

            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch])
        y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch])

        yield x_batch, y_batch


def get_batch_val(ex_path, batch_size, patch_size):
    data = [None] * 4
    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0].split('_')[-1]
        image = im_path_to_arr(im_path)
        if im_type == 't1':
            data[0] = normalize_image(image)
        if im_type == 't1c' or im_type == 't1ce':
            data[1] = normalize_image(image)
        if im_type == 't2':
            data[2] = normalize_image(image)
        if im_type == 'flair' or im_type == 'fla':
            data[3] = normalize_image(image)
        if im_type == 'tumor' or im_type == 'seg':
            labels = preprocess_labels(image)

    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    i_batch = []
    j_batch = []
    k_batch = []
    x_batch = []
    y_batch = []

    i_len, j_len, k_len = labels.shape

    i_rem = i_len % patch_size
    j_rem = j_len % patch_size
    k_rem = k_len % patch_size

    batch_count = 0

    for i in range(i_rem // 2, i_len, patch_size):
        for j in range(j_rem // 2, j_len, patch_size):
            for k in range(k_rem // 2, k_len, patch_size):

                if (i + patch_size) <= i_len and (j + patch_size) <= j_len and (k + patch_size) <= k_len:

                    x = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
                    y = labels[i:i + patch_size, j:j + patch_size, k:k + patch_size]

                    i_batch.append(i)
                    j_batch.append(j)
                    k_batch.append(k)
                    x_batch.append(x)
                    y_batch.append(y)

                    batch_count += 1

                    if batch_count == batch_size:
                        x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch])
                        y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch])
                        yield i_batch, j_batch, k_batch, x_batch, y_batch

                        i_batch = []
                        j_batch = []
                        k_batch = []
                        x_batch = []
                        y_batch = []

                        batch_count = 0

    if batch_count != 0:
        x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch])
        y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch])
        yield i_batch, j_batch, k_batch, x_batch, y_batch


def create_dataset_placeholder():

train_batch_size = self.config.train_batch_size

image_batch_input = tf.placeholder(tf.float32, shape=[train_batch_size, 155, 240, 240, 4])
label_batch_input = tf.placeholder(tf.int32, shape=[train_batch_size, 155, 240, 240])


# q = tf.RandomShuffleQueue(capacity=100, min_after_dequeue=1,
#                           dtypes=[tf.float32, tf.int32],
#                           shapes=[[None, 155, 240, 240, 4], [None, 155, 240, 240]])
q = tf.FIFOQueue(100, [tf.float32, tf.int32],
                 shapes=[[train_batch_size, 155, 240, 240, 4], [train_batch_size, 155, 240, 240]])
enqueue_op = q.enqueue_many([image_batch_input, label_batch_input])


image_batch, label_batch = q.dequeue_many(BATCH_SIZE)


sess = tf.Session()

def load_and_enqueue():

    with open(...) as feature_file, open(...) as label_file:
        while True:
            feature_array = numpy.fromfile(feature_file, numpy.float32, 100)
            if not feature_array:
                return
            label_value = numpy.fromfile(feature_file, numpy.int32, 1)[0]

            sess.run(enqueue_op, feed_dict={image_batch_input: feature_array,
                                            label_batch_input: label_value})

# Start a thread to enqueue data asynchronously, and hide I/O latency.
t = threading.Thread(target=load_and_enqueue)
t.start()

for _ in range(TRAINING_EPOCHS):
  sess.run(train_op)
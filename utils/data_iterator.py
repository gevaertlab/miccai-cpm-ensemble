import os

import numpy as np

from utils.data_utils import im_path_to_arr


def data_iter(ex_path, samp_mode, batch_size, num_batches):
    """ Generate input and label data from the BRATS dataset.

    params:
    ex_path: path to example directory
    samp_mode: the sampling algorithm used to extract image patches ('fgbg', 'unif')
    batch_size: the number of image patches per batch
    num_batches: the number of batches per image

    Yields patches of data and labels as inputs to a dense DNN classifier.
    The sampling algorithm 'fgbg' selects patches uniformly randomly from the
    foreground class with probability 0.5 or uniformly randomly from the
    background class with probability 0.5. The sampling algorithm 'unif' selects
    patches uniformly randomly from the entire the brain region. The sampling
    algorithm 'full' selects patches equally spaced to cover the entire MRI
    image for generating full segmentations.

    yields:
    x: data patch (batch_sizex25x25x25x4)
    y: labels patch (batch_sizex9x9x9)
    """
    data = [None] * 4

    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0]
        if im_type == 't1':
            data[0] = im_path_to_arr(im_path)
        if im_type == 't1c':
            data[1] = im_path_to_arr(im_path)
        if im_type == 't2':
            data[2] = im_path_to_arr(im_path)
        if im_type == 'flair':
            data[3] = im_path_to_arr(im_path)
        if im_type == 'tumor':
            labels = im_path_to_arr(im_path)

    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    if samp_mode == 'fgbg':

        trimmed_data = data[12:-12, 12:-12, 12:-12, :]
        trimmed_labels = labels[12:-12, 12:-12, 12:-12]

        bg = np.where((trimmed_labels == 0) & (trimmed_data[..., 0] != 0))
        fg = np.where((trimmed_labels == 1) & (trimmed_data[..., 0] != 0))

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
                    x = data[i:i + 25, j:j + 25, k:k + 25, :]
                    y = labels[i + 8:i + 17, j + 8:j + 17, k + 8:k + 17]
                else:
                    idx = np.random.randint(num_fg)
                    i = fg[0][idx]
                    j = fg[1][idx]
                    k = fg[2][idx]
                    x = data[i:i + 25, j:j + 25, k:k + 25, :]
                    y = labels[i + 8:i + 17, j + 8:j + 17, k + 8:k + 17]

                x_batch.append(x)
                y_batch.append(y)

            x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch])
            y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch])

            yield x_batch, y_batch

    elif samp_mode == 'full':

        i_batch = []
        j_batch = []
        k_batch = []
        x_batch = []
        y_batch = []

        batch_count = 0

        i_len, j_len, k_len = labels.shape

        for i in get_patch_centers(i_len):
            for j in get_patch_centers(j_len):
                for k in get_patch_centers(k_len):

                    x = data[i - 12:i + 13, j - 12:j + 13, k - 12:k + 13, :]
                    y = labels[i - 4:i + 5, j - 4:j + 5, k - 4:k + 5]

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

# generalize this later
def get_patch_centers(im_size):
    quotient = (im_size - 16) // 9
    remainder = (im_size - 16) % 9
    start = remainder // 2 + 12
    end = start + (quotient - 1) * 9 + 1
    return range(start, end, 9)



def fcn_data_iter(ex_path, samp_mode, batch_size, num_batches, patch_size):
    """ Generate input and label data from the BRATS dataset.

    params:
    ex_path: path to example directory
    samp_mode: the sampling algorithm used to extract image patches ('fgbg', 'unif')
    batch_size: the number of image patches per batch
    num_batches: the number of batches per image

    Yields patches of data and labels as inputs to a dense DNN classifier.
    The sampling algorithm 'fgbg' selects patches uniformly randomly from the
    foreground class with probability 0.5 or uniformly randomly from the
    background class with probability 0.5. The sampling algorithm 'unif' selects
    patches uniformly randomly from the entire the brain region. The sampling
    algorithm 'full' selects patches equally spaced to cover the entire MRI
    image for generating full segmentations.

    yields:
    x: data patch (batch_sizex25x25x25x4)
    y: labels patch (batch_sizex9x9x9)
    """
    data = [None] * 4
    half_patch = patch_size // 2

    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0]
        if im_type == 't1':
            data[0] = im_path_to_arr(im_path)
        if im_type == 't1c':
            data[1] = im_path_to_arr(im_path)
        if im_type == 't2':
            data[2] = im_path_to_arr(im_path)
        if im_type == 'flair':
            data[3] = im_path_to_arr(im_path)
        if im_type == 'tumor':
            labels = im_path_to_arr(im_path)

    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    if samp_mode == 'fgbg':

        # trimmed_data = data[0:-patch_size, 0:-patch_size, 0:-patch_size, :]
        # trimmed_labels = labels[0:-patch_size, 0:-patch_size, 0:-patch_size]

        trimmed_data = data[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch, :]
        trimmed_labels = labels[half_patch:-half_patch, half_patch:-half_patch, half_patch:-half_patch]

        # print(data.shape)
        # print(trimmed_data.shape)

        bg = np.where((trimmed_labels == 0) & (trimmed_data[..., 0] != 0))
        fg = np.where((trimmed_labels == 1) & (trimmed_data[..., 0] != 0))

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
                    # x = data[i-half_patch:i+half_patch, j-half_patch:j+half_patch, k-half_patch:k+half_patch, :]
                    # y = labels[i-half_patch:i+half_patch, j-half_patch:j+half_patch, k-half_patch:k+half_patch]
                    x = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
                    y = labels[i:i + patch_size, j:j + patch_size, k:k + patch_size]
                    # print(x.shape)
                else:
                    idx = np.random.randint(num_fg)
                    i = fg[0][idx]
                    j = fg[1][idx]
                    k = fg[2][idx]
                    x = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
                    y = labels[i:i + patch_size, j:j + patch_size, k:k + patch_size]
                    # print(x.shape)

                x_batch.append(x)
                y_batch.append(y)

            x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch])
            y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch])

            yield x_batch, y_batch

    elif samp_mode == 'full':

        i_batch = []
        j_batch = []
        k_batch = []
        x_batch = []
        y_batch = []

        batch_count = 0

        i_len, j_len, k_len = labels.shape

        i_rem = i_len % patch_size
        j_rem = j_len % patch_size
        k_rem = k_len % patch_size

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

                    else:
                        continue

        if batch_count != 0:
            x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch])
            y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch])
            yield i_batch, j_batch, k_batch, x_batch, y_batch



def heidelberg_iter(ex_path, samp_mode, batch_size, num_batches):
    """ Generate input and label data for the Heidelberg experiments.

    params:
    ex_path: path to example directory
    samp_mode: the sampling algorithm used to extract image patches ('fgbg', 'unif')
    batch_size: the number of image patches per batch
    num_batches: the number of batches per image

    Yields patches of data and labels as inputs to a dense DNN classifier.
    The sampling algorithm 'fgbg' selects patches uniformly randomly from the
    foreground class with probability 0.5 or uniformly randomly from the
    background class with probability 0.5. The sampling algorithm 'full' selects
    patches equally spaced to cover the entire MRI image for generating full
    segmentations.

    yields:
    x: data patch (batch_sizex25x25x25x4)
    y: labels patch (batch_sizex9x9x9)
    """
    data = [None]*5

    for im_name in os.listdir(ex_path):
        im_path = os.path.join(ex_path, im_name)
        im_type = im_name.split('.')[0]
        if im_type == 't1':
            data[0] = im_path_to_arr(im_path)
        if im_type == 't1c':
            data[1] = im_path_to_arr(im_path)
        if im_type == 't1c_sub':
            data[2] = im_path_to_arr(im_path)
        if im_type == 't2':
            data[3] = im_path_to_arr(im_path)
        if im_type == 'flair':
            data[4] = im_path_to_arr(im_path)
        if im_type == 'seg':
            labels = im_path_to_arr(im_path)

    data = np.concatenate([item[..., np.newaxis] for item in data], axis=3)

    if samp_mode == 'fgbg':

        trimmed_data = data[12:-12, 12:-12, 12:-12, :]
        trimmed_labels = labels[12:-12, 12:-12, 12:-12]

        bg = np.where((trimmed_labels == 0) & (trimmed_data[..., 0] != 0))
        fg = np.where((trimmed_labels == 1) & (trimmed_data[..., 0] != 0))

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
                    x = data[i:i+25, j:j+25, k:k+25, :]
                    y = labels[i+8:i+17, j+8:j+17, k+8:k+17]
                else:
                    idx = np.random.randint(num_fg)
                    i = fg[0][idx]
                    j = fg[1][idx]
                    k = fg[2][idx]
                    x = data[i:i+25, j:j+25, k:k+25, :]
                    y = labels[i+8:i+17, j+8:j+17, k+8:k+17]

                x_batch.append(x)
                y_batch.append(y)

            x_batch = np.concatenate([item[np.newaxis, ...] for item in x_batch])
            y_batch = np.concatenate([item[np.newaxis, ...] for item in y_batch])

            yield x_batch, y_batch

    elif samp_mode == 'full':

        i_batch = []
        j_batch = []
        k_batch = []
        x_batch = []
        y_batch = []

        batch_count = 0

        i_len, j_len, k_len = labels.shape

        for i in get_patch_centers(i_len):
            for j in get_patch_centers(j_len):
                for k in get_patch_centers(k_len):

                    x = data[i-12:i+13, j-12:j+13, k-12:k+13, :]
                    y = labels[i-4:i+5, j-4:j+5, k-4:k+5]

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

from utils.config import Config

directory = '/local-scratch/romain_scratch/brats2017/val'
config = Config('config_files/fcn_train_concat_2017_v28.cfg')
dataset = get_dataset_v2('/local-scratch/romain_scratch/brats2017/val/', False, config, 'Brats')

patch_size = config.patch_size
iterator = tf.contrib.data.Iterator.from_structure(\
                output_types=(tf.string, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32),
                output_shapes=([None], [None], [None], [None],\
                               [None, patch_size, patch_size, patch_size, 4],\
                               [None, patch_size, patch_size, patch_size]))
p, i, j, k, im, lab = iterator.get_next()
init_op = iterator.make_initializer(dataset)

with tf.Session() as sess:
    sess.run(init_op)
    d = sess.run(p)
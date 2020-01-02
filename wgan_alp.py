import argparse
import datetime
from functools import partial
import os
import numpy as np
import pickle
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# DATA
class ImageDataSet(object):
    def __init__(self, images):
        assert images.ndim == 4

        self.num_examples = images.shape[0]

        self.images = images
        self.epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        assert batch_size <= self.num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.images[start:end]


def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct


def get_cifar10_dataset(split=None):
    train_dir = "/cache/data/cifar10/"

    data = []
    for i in range(1, 7):
        if i < 6:
            path = os.path.join(train_dir, 'cifar-10-batches-py', 'data_batch_{}'.format(i))
        elif i == 6:
            path = os.path.join(train_dir, 'cifar-10-batches-py', 'test_batch')
        dct = unpickle(path)
        data.append(dct[b'data'])

    data_arr = np.concatenate(data, axis=0)
    raw_float = np.array(data_arr, dtype='float32') / 256.0
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])

    if split is None:
        pass
    elif split == 'train':
        images = images[:-10000]
    elif split == 'test':
        images = images[-10000:]
    else:
        raise ValueError('unknown split')

    dataset = ImageDataSet(images)

    return dataset


def random_flip(x, up_down=False, left_right=True):
    with tf.name_scope('random_flip'):
        s = tf.shape(x)
        if up_down:
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[1]))
        if left_right:
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[2]))
        return x


def get_cifar10_tf(batch_size=1, shape=[32, 32], split=None, augment=True, start_queue_runner=True):
    with tf.name_scope('get_cifar10_tf'):
        dataset = get_cifar10_dataset(split=split)

        images = tf.constant(dataset.images, dtype='float32')

        image = tf.train.slice_input_producer([images], shuffle=True)

        images_batch = tf.train.batch(image, batch_size=batch_size, num_threads=8)

        if augment:
            images_batch = random_flip(images_batch)
            images_batch += tf.random_uniform(tf.shape(images_batch), 0.0, 1.0/256.0)

        if shape != [32, 32]:
            images_batch = tf.image.resize_bilinear(images_batch, [shape[0], shape[1]])

        if start_queue_runner:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

        return images_batch


def image_grid(x, size=8):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image[None]


def image_grid_summary(name, x):
    with tf.name_scope(name):
        tf.summary.image('grid', image_grid(x))


def scalars_summary(name, x):
    with tf.name_scope(name):
        x = tf.reshape(x, [-1])
        mean, var = tf.nn.moments(x, axes=0)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('std', tf.sqrt(var))


def apply_conv(x, filters=32, kernel_size=3, he_init=True):
    if he_init:
        initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)
    else:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    return tf.layers.conv2d(
        x, filters=filters, kernel_size=kernel_size, padding='SAME', kernel_initializer=initializer
    )


def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.relu(x)


def bn(x):
    return tf.contrib.layers.batch_norm(
        x,
        decay=0.9,
        center=True,
        scale=True,
        epsilon=1e-5,
        zero_debias_moving_mean=True,
        is_training=is_training
    )


def stable_norm(x, ord):
    return tf.norm(tf.contrib.layers.flatten(x), ord=ord, axis=1, keepdims=True)


def normalize(x, ord):
    return x / tf.maximum(tf.expand_dims(tf.expand_dims(stable_norm(x, ord=ord), -1), -1), 1e-10)


def downsample(x):
    with tf.name_scope('downsample'):
        x = tf.identity(x)
        return tf.add_n(
            [x[:,::2,::2,:], x[:,1::2,::2,:], x[:,::2,1::2,:], x[:,1::2,1::2,:]]
        ) / 4


def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        x = tf.concat([x, x, x, x], axis=-1)
        return tf.depth_to_space(x, 2)


def conv_meanpool(x, **kwargs):
    return downsample(apply_conv(x, **kwargs))


def meanpool_conv(x, **kwargs):
    return apply_conv(downsample(x), **kwargs)


def upsample_conv(x, **kwargs):
    return apply_conv(upsample(x), **kwargs)


def resblock(x, filters, resample=None, normalize=False):
    if normalize:
        norm_fn = bn
    else:
        norm_fn = tf.identity

    if resample == 'down':
        conv_1 = partial(apply_conv, filters=filters)
        conv_2 = partial(conv_meanpool, filters=filters)
        conv_shortcut = partial(conv_meanpool, filters=filters, kernel_size=1, he_init=False)
    elif resample == 'up':
        conv_1 = partial(upsample_conv, filters=filters)
        conv_2 = partial(apply_conv, filters=filters)
        conv_shortcut = partial(upsample_conv, filters=filters, kernel_size=1, he_init=False)
    elif resample == None:
        conv_1 = partial(apply_conv, filters=filters)
        conv_2 = partial(apply_conv, filters=filters)
        conv_shortcut = tf.identity

    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = conv_1(activation(norm_fn(x)))
        update = conv_2(activation(norm_fn(update)))

        skip = conv_shortcut(x)
        return skip + update


def resblock_optimized(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = conv_meanpool(activation(update), filters=filters)
        # update = conv_meanpool(activation(bn(update)), filters=filters)

        skip = meanpool_conv(x, filters=filters, kernel_size=1, he_init=False)
        return skip + update


def generator(z, reuse):
    with tf.variable_scope('generator', reuse=reuse):
        channels = 128
        with tf.name_scope('pre_process'):
            z = tf.layers.dense(z, 4 * 4 * channels)
            x = tf.reshape(z, [-1, 4, 4, channels])

        with tf.name_scope('x1'):
            x = resblock(x, filters=channels, resample='up', normalize=True) # 8
            x = resblock(x, filters=channels, resample='up', normalize=True) # 16
            x = resblock(x, filters=channels, resample='up', normalize=True) # 32

        with tf.name_scope('post_process'):
            x = activation(bn(x))
            result = apply_conv(x, filters=3, he_init=False)
            return tf.tanh(result)


def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.name_scope('pre_process'):
            x = resblock_optimized(x, filters=128)

        with tf.name_scope('x1'):
            x = resblock(x, filters=128, resample='down') # 8
            x = resblock(x, filters=128) # 16
            x = resblock(x, filters=128) # 32
            # x = resblock(x, filters=128, resample='down', normalize=True) # 8
            # x = resblock(x, filters=128, normalize=True) # 16
            # x = resblock(x, filters=128, normalize=True) # 32

        with tf.name_scope('post_process'):
            x = activation(x)
            # x = activation(bn(x))
            x = tf.reduce_mean(x, axis=[1, 2])
            flat = tf.contrib.layers.flatten(x)
            flat = tf.layers.dense(flat, 1)
            return flat


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",                   default="/cache/logs/wgan_alp/")
    parser.add_argument("--log_freq",      type=int,   default=1)
    parser.add_argument("--iterations",    type=int,   default=100000)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--save_freq",     type=int,   default=-1)
    parser.add_argument("--val_freq",      type=int,   default=1000)
    parser.add_argument("--val_size",      type=int,   default=100)
    parser.add_argument("--random_seed",   type=int,   default=0)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--b1",            type=float, default=0.0)
    parser.add_argument("--b2",            type=float, default=0.9)
    parser.add_argument("--latent_dim",    type=int,   default=128)
    parser.add_argument("--lambda_lp",     type=float, default=10)
    parser.add_argument("--eps_min",       type=float, default=0.1)
    parser.add_argument("--eps_max",       type=float, default=10.0)
    parser.add_argument("--xi",            type=float, default=10.0)
    parser.add_argument("--ip",            type=int,   default=1)
    parser.add_argument("--K",             type=float, default=1)
    parser.add_argument("--p",             type=float, default=2)
    parser.add_argument("--n_critic",      type=int,   default=5)
    parser.add_argument("--reduce_fn",                 default="mean", choices=["mean", "sum", "max"])
    parser.add_argument("--reg",                       default="alp", choices=["gp", "lp", "alp"])
    args = parser.parse_args()
    print(args)

    # set seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    sess = tf.InteractiveSession()

    run_name = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    log_dir = args.log_dir + run_name
    os.makedirs(log_dir)

    reduce_fn = {
        "mean": tf.reduce_mean,
        "sum": tf.reduce_sum,
        "max": tf.reduce_max,
    }[args.reduce_fn]

    with tf.name_scope('placeholders'):
        x_train_ph = get_cifar10_tf(batch_size=args.batch_size)
        x_test_ph = get_cifar10_tf(batch_size=args.val_size)
        x_10k_ph = get_cifar10_tf(batch_size=10000)
        x_50k_ph = get_cifar10_tf(batch_size=50000)

        is_training = tf.placeholder(bool, name='is_training')
        use_agumentation = tf.identity(is_training, name='is_training')

    with tf.name_scope('pre_process'):
        x_train = (x_train_ph - 0.5) * 2.0
        x_test = (x_test_ph - 0.5) * 2.0

        x_true = tf.cond(is_training, lambda: x_train, lambda: x_test)

        x_10k = (x_10k_ph - 0.5) * 2.0
        x_50k = (x_50k_ph - 0.5) * 2.0

    with tf.name_scope('gan'):
        z = tf.random_normal([tf.shape(x_true)[0], 128], name="z")

        x_generated = generator(z, reuse=False)

        d_true = discriminator(x_true, reuse=False)
        d_generated = discriminator(x_generated, reuse=True)

        z_gen = tf.random_normal([args.batch_size * 2, 128], name="z")
        d_generated_train = discriminator(generator(z_gen, reuse=True), reuse=True)

    with tf.name_scope('regularizer'):
        epsilon = tf.random_uniform([tf.shape(x_true)[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x_generated + (1 - epsilon) * x_true
        d_hat = discriminator(x_hat, reuse=True)

        gradients = tf.gradients(d_hat, x_hat)[0]

        dual_p = 1 / (1 - 1 / args.p) if args.p != 1 else np.inf
        gradient_norms = stable_norm(gradients, ord=dual_p)

        gp = gradient_norms - args.K
        gp_loss = args.lambda_lp * reduce_fn(gp ** 2)

        lp = tf.maximum(gradient_norms - args.K, 0)
        lp_loss = args.lambda_lp * reduce_fn(lp ** 2)

    with tf.name_scope('alp'):
        samples = tf.concat([x_true, x_generated], axis=0)

        eps = args.eps_min + (args.eps_max - args.eps_min) * tf.random_uniform([tf.shape(samples)[0], 1, 1, 1], 0, 1)

        validity = discriminator(samples, reuse=True)

        d = tf.random_uniform(tf.shape(samples), 0, 1) - 0.5
        d = normalize(d, ord=2)
        for _ in range(args.ip):
            samples_hat = tf.clip_by_value(samples + args.xi * d, clip_value_min=-1, clip_value_max=1)
            validity_hat = discriminator(samples_hat, reuse=True)
            dist = tf.reduce_mean(tf.abs(validity - validity_hat))
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = normalize(tf.stop_gradient(grad), ord=2)
        r_adv = d * eps

        samples_hat = tf.clip_by_value(samples + r_adv, clip_value_min=-1, clip_value_max=1)

        d_lp                   = lambda x, x_hat: stable_norm(x - x_hat, ord=args.p)
        d_x                    = d_lp

        samples_diff = d_x(samples, samples_hat)
        samples_diff = tf.maximum(samples_diff, 1e-10)

        validity      = discriminator(samples    , reuse=True)
        validity_hat  = discriminator(samples_hat, reuse=True)
        validity_diff = tf.abs(validity - validity_hat)

        alp = tf.maximum(validity_diff / samples_diff - args.K, 0)
        # alp = tf.abs(validity_diff / samples_diff - args.K)

        nonzeros = tf.greater(alp, 0)
        count = tf.reduce_sum(tf.cast(nonzeros, tf.float32))

        alp_loss = args.lambda_lp * reduce_fn(alp ** 2)

    with tf.name_scope('loss_gan'):
        wasserstein = (tf.reduce_mean(d_generated) - tf.reduce_mean(d_true))

        g_loss = tf.reduce_mean(d_generated_train)
        d_loss = -wasserstein
        if args.reg == 'gp':
            d_loss += gp_loss
        elif args.reg == 'lp':
            d_loss += lp_loss
        elif args.reg == 'alp':
            d_loss += alp_loss

    with tf.name_scope('optimizer'):

        global_step = tf.Variable(0, trainable=False, name='global_step')
        decay = tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / args.iterations))
        learning_rate = args.lr * decay
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/generator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        with tf.control_dependencies(update_ops):
            g_train = optimizer.minimize(g_loss, var_list=g_vars, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/discriminator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        with tf.control_dependencies(update_ops):
            d_train = optimizer.minimize(d_loss, var_list=d_vars)

    with tf.name_scope('summaries'):
        tf.summary.scalar('wasserstein', wasserstein)

        tf.summary.scalar('g_loss', g_loss)

        tf.summary.scalar('d_loss', d_loss)
        scalars_summary('d_true', d_true)
        scalars_summary('d_generated', d_generated)
        tf.summary.scalar('gp_loss', gp_loss)
        tf.summary.scalar('lp_loss', lp_loss)

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('global_step', global_step)

        scalars_summary('x_generated', x_generated)
        scalars_summary('x_true', x_true)

        image_grid_summary('x_true', x_true)
        image_grid_summary('x_generated', x_generated)
        image_grid_summary('gradients', gradients)

        scalars_summary('gradient_norms', gradient_norms)
        scalars_summary('gradients', gradients)

        tf.summary.scalar('alp_loss', alp_loss)
        tf.summary.scalar('count', count)
        scalars_summary('alp', alp)

        merged_summary = tf.summary.merge_all()

        # Advanced metrics
        with tf.name_scope('validation'):
            # INCEPTION VALIDATION
            # Specific function to compute inception score for very large number of samples
            def generate_resize_and_classify(z):
                INCEPTION_OUTPUT = 'logits:0'
                x = generator(z, reuse=True)
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_OUTPUT)

            # Fixed z for fairness between runs
            inception_z = tf.constant(np.random.randn(10000, 128), dtype='float32')
            inception_score = tf.contrib.gan.eval.classifier_score(
                inception_z,
                classifier_fn=generate_resize_and_classify,
                num_batches=10000 // 100
            )

            inception_summary = tf.summary.merge([
                tf.summary.scalar('inception_score', inception_score)
            ])

            # FID VALIDATION
            def resize_and_classify(x):
                INCEPTION_FINAL_POOL = 'pool_3:0'
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_FINAL_POOL)

            fid_real = x_10k
            fid_z = tf.constant(np.random.randn(10000, 128), dtype='float32')
            fid_z_list = array_ops.split(fid_z, num_or_size_splits=10000 // 100)
            fid_z_batches = array_ops.stack(fid_z_list)
            fid_gen = functional_ops.map_fn(
                fn=partial(generator, reuse=True),
                elems=fid_z_batches,
                parallel_iterations=1,
                back_prop=False,
                swap_memory=True,
                name='RunGenerator'
            )
            fid_gen = array_ops.concat(array_ops.unstack(fid_gen), 0)
            fid = tf.contrib.gan.eval.frechet_classifier_distance(
                fid_real,
                fid_gen,
                classifier_fn=resize_and_classify,
                num_batches=10000 // 100
            )

            fid_summary = tf.summary.merge([
                tf.summary.scalar('fid', fid)
            ])

            full_summary = tf.summary.merge([merged_summary, inception_summary, fid_summary])

        # Final eval
        with tf.name_scope('test'):
            # INCEPTION TEST
            # Specific function to compute inception score for very large number of samples
            def generate_resize_and_classify(z):
                INCEPTION_OUTPUT = 'logits:0'
                x = generator(z, reuse=True)
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_OUTPUT)


            # Fixed z for fairness between runs
            inception_z_final = tf.constant(np.random.randn(100000, 128), dtype='float32')
            inception_score_final = tf.contrib.gan.eval.classifier_score(
                inception_z_final,
                classifier_fn=generate_resize_and_classify,
                num_batches=100000 // 100
            )

            inception_summary_final = tf.summary.merge([
                tf.summary.scalar('inception_score_final', inception_score_final)
            ])

            # FID TEST
            def resize_and_classify(x):
                INCEPTION_FINAL_POOL = 'pool_3:0'
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_FINAL_POOL)

            fid_real_final = x_50k
            fid_z_final = tf.constant(np.random.randn(50000, 128), dtype='float32')
            fid_z_final_list = array_ops.split(fid_z_final, num_or_size_splits=50000 // 100)
            fid_z_final_batches = array_ops.stack(fid_z_final_list)
            fid_gen_final = functional_ops.map_fn(
                fn=partial(generator, reuse=True),
                elems=fid_z_final_batches,
                parallel_iterations=1,
                back_prop=False,
                swap_memory=True,
                name='RunGenerator'
            )
            fid_gen_final = array_ops.concat(array_ops.unstack(fid_gen_final), 0)
            fid_final = tf.contrib.gan.eval.frechet_classifier_distance(
                fid_real_final,
                fid_gen_final,
                classifier_fn=resize_and_classify,
                num_batches=50000 // 100
            )

            fid_summary_final = tf.summary.merge([
                tf.summary.scalar('fid_final', fid_final)
            ])

            final_summary = tf.summary.merge([merged_summary, inception_summary_final, fid_summary_final])

        summary_writer = tf.summary.FileWriter(log_dir)

    # Initialize all TF variables
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ])

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Add op to save and restore
    saver = tf.train.Saver()

    # Standardized validation z
    z_validate = np.random.randn(args.val_size, 128)

    print(f"Logging to: {log_dir}")

    # Train the network
    t = tqdm(range(args.iterations))
    for _ in t:
        i = sess.run(global_step)

        for j in range(args.n_critic):
            results = sess.run(
                [d_train, d_loss],
                feed_dict={is_training: True}
            )
            d_loss_result = results[1]

        _, g_loss_result = sess.run(
            [g_train, g_loss],
            feed_dict={is_training: True}
        )

        if i % args.log_freq == args.log_freq - 1:
            merged_summary_result_train = sess.run(
                merged_summary,
                feed_dict={is_training: False}
            )
            summary_writer.add_summary(merged_summary_result_train, i)
        if i % args.val_freq == args.val_freq - 1:
            merged_summary_result_test = sess.run(
                full_summary,
                feed_dict={is_training: False}
            )
            summary_writer.add_summary(merged_summary_result_test, i)

        if i % args.save_freq == args.save_freq - 1:
            saver.save(sess, log_dir)

        t.set_description(
            f"[Iteration {i}/{args.iterations} [D loss: {d_loss_result}] [G loss: {g_loss_result}]]"
        )

        if (i + 1) == args.iterations:
            merged_summary_result_final = sess.run(
                final_summary,
                feed_dict={is_training: False}
            )
            summary_writer.add_summary(merged_summary_result_final, args.iterations)


import tensorflow as tf

def sel_images(images, iter_num, t_fullrange, sequence_length, vidpred_data=False):
    delta_t = tf.cast(tf.ceil(sequence_length * (tf.cast(iter_num + 1, tf.float32)) / t_fullrange), tf.int32)
    delta_t = tf.clip_by_value(delta_t, 1, sequence_length - 1)
    tstart = tf.random_uniform([1], 0, sequence_length - delta_t, dtype=tf.int32)
    tend =   tstart + tf.random_uniform([1], tf.ones([], dtype=tf.int32), delta_t + 1, dtype=tf.int32)
    begin = tf.stack([0, tf.squeeze(tstart), 0, 0, 0], 0)

    I0 = tf.squeeze(tf.slice(images, begin, [-1, 1, -1, -1, -1]))
    begin = tf.stack([0, tf.squeeze(  tend), 0, 0, 0], 0)
    I1 = tf.squeeze(tf.slice(images, begin, [-1, 1, -1, -1, -1]))
    return I0, I1
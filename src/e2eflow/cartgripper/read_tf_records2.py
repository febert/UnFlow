import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import pdb
import time

from PIL import Image
import imp

import _pickle as cPickle


def build_tfrecord_input(conf, training=True, input_file=None):
    """Create input tfrecord tensors.

    Args:
      training: training or validation data_files.
      conf: A dictionary containing the configuration for the experiment
    Returns:
      list of tensors corresponding to images, actions, and states. The images
      tensor is 5D, batch x time x height x width x channels. The state and
      action tensors are 3D, batch x time x dimension.
    Raises:
      RuntimeError: if no files found.
    """
    if 'sdim' in conf:
        sdim = conf['sdim']
    else: sdim = 3
    if 'adim' in conf:
        adim = conf['adim']
    else: adim = 4
    print('adim', adim)
    print('sdim', sdim)

    if input_file is not None:
        filenames = [input_file]
        shuffle = False
    else:
        filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
        if not filenames:
            raise RuntimeError('No data_files files found.')

        index = int(np.floor(conf['train_val_split'] * len(filenames)))
        if training:
            filenames = filenames[:index]
        else:
            filenames = filenames[index:]

        if conf['visualize']:  #if visualize do not perform train val split
            filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
            shuffle = False
        else:
            shuffle = True

    print('using shuffle: ', shuffle)

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    image_seq, image_main_seq, endeffector_pos_seq, gen_images_seq, gen_states_seq,\
    action_seq, object_pos_seq, robot_pos_seq = [], [], [], [], [], [], [], []

    load_indx = range(0, conf['sequence_length'], conf['skip_frame'])
    print('using frame sequence: ', load_indx)

    rand_h = tf.random_uniform([1], minval=-0.2, maxval=0.2)
    rand_s = tf.random_uniform([1], minval=-0.2, maxval=0.2)
    rand_v = tf.random_uniform([1], minval=-0.2, maxval=0.2)

    for i in load_indx:

        image_name = str(i) + '/image_view0/encoded'

        if 'image_only' not in conf:
            action_name = str(i) + '/action'
            endeffector_pos_name = str(i) + '/endeffector_pos'

        features = {
            image_name: tf.FixedLenFeature([1], tf.string),
        }

        if 'image_only' not in conf:
            features[action_name] = tf.FixedLenFeature([adim], tf.float32)
            features[endeffector_pos_name] = tf.FixedLenFeature([sdim], tf.float32)

        if 'test_metric' in conf:
            robot_pos_name = str(i) + '/robot_pos'
            object_pos_name = str(i) + '/object_pos'
            features[robot_pos_name] = tf.FixedLenFeature([conf['test_metric']['robot_pos'] * 2], tf.int64)
            features[object_pos_name] = tf.FixedLenFeature([conf['test_metric']['object_pos'] * 2], tf.int64)

        if 'load_vidpred_data' in conf:
            gen_image_name = str(i) + '/gen_images'
            gen_states_name = str(i) + '/gen_states'
            features[gen_image_name] = tf.FixedLenFeature([1], tf.string)
            features[gen_states_name] = tf.FixedLenFeature([sdim], tf.float32)

        features = tf.parse_single_example(serialized_example, features=features)

        COLOR_CHAN = 3

        if 'orig_size' in conf:
            ORIGINAL_HEIGHT = conf['orig_size'][0]
            ORIGINAL_WIDTH = conf['orig_size'][1]
        else:
            ORIGINAL_WIDTH = 64
            ORIGINAL_HEIGHT = 64

        if 'row_start' in conf:
            IMG_HEIGHT = conf['row_end'] - conf['row_start']
        else:
            IMG_HEIGHT = ORIGINAL_HEIGHT

        if 'img_width' in conf:
            IMG_WIDTH = conf['img_width']
        else:
            IMG_WIDTH = ORIGINAL_WIDTH

        image = tf.decode_raw(features[image_name], tf.uint8)
        image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
        image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
        if 'row_start' in conf:
            image = image[conf['row_start']:conf['row_end']]
        image = tf.reshape(image, [1, IMG_HEIGHT, IMG_WIDTH, COLOR_CHAN])

        image = tf.cast(image, tf.float32) / 255.0

        if 'color_augmentation' in conf:
            # print 'performing color augmentation'
            image_hsv = tf.image.rgb_to_hsv(image)
            img_stack = [tf.unstack(imag, axis=2) for imag in tf.unstack(image_hsv, axis=0)]
            stack_mod = [tf.stack([x[0] + rand_h,
                                   x[1] + rand_s,
                                   x[2] + rand_v]
                                  , axis=2) for x in img_stack]

            image_rgb = tf.image.hsv_to_rgb(tf.stack(stack_mod))
            image = tf.clip_by_value(image_rgb, 0.0, 1.0)

        image_seq.append(image)

        if 'image_only' not in conf:
            endeffector_pos = tf.reshape(features[endeffector_pos_name], shape=[1, sdim])
            endeffector_pos_seq.append(endeffector_pos)
            action = tf.reshape(features[action_name], shape=[1, adim])
            action_seq.append(action)

        if 'test_metric' in conf:
            robot_pos = tf.reshape(features[robot_pos_name], shape=[1, 2])
            robot_pos_seq.append(robot_pos)

            object_pos = tf.reshape(features[object_pos_name], shape=[1, conf['test_metric']['object_pos'], 2])
            object_pos_seq.append(object_pos)

        if 'load_vidpred_data' in conf:
            gen_images = tf.decode_raw(features[gen_image_name], tf.uint8)
            gen_images = tf.reshape(gen_images, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
            gen_images = tf.reshape(gen_images, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
            gen_images = tf.reshape(gen_images, [1, IMG_HEIGHT, IMG_WIDTH, COLOR_CHAN])
            gen_images = tf.cast(gen_images, tf.float32) / 255.0
            gen_images_seq.append(gen_images)

            gen_states = tf.reshape(features[gen_states_name], shape=[1, sdim])
            gen_states_seq.append(gen_states)

    image_seq = tf.concat(values=image_seq, axis=0)

    # endeffector_pos_seq = tf.concat(endeffector_pos_seq, 0)
    # action_seq = tf.concat(action_seq, 0)

    image_batch = tf.train.batch(
        [image_seq],
        conf['batch_size'],
        num_threads=10,
        capacity=100 * conf['batch_size'])
    return image_batch


def main():
    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = '/mnt/sda1/pushing_data/cartgripper_startgoal_large4step/train'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 4 #48      # 'sequence length, including context frames.'
    conf['batch_size']= 10
    conf['visualize']= True
    conf['context_frames'] = 2

    # conf['row_start'] = 15
    # conf['row_end'] = 63

    conf['sdim'] = 6
    conf['adim'] = 3

    conf['image_only'] = ''

    # conf['orig_size'] = [48, 64]
    conf['orig_size'] = [480, 640]
    # conf['load_vidpred_data'] = ''

    # conf['color_augmentation'] = ''
    # conf['test_metric'] = {'robot_pos': 1, 'object_pos': 2}

    print ('-------------------------------------------------------------------')
    print ('verify current settings!! ')
    for key in conf.keys():
        print (key, ': ', conf[key])
    print ('-------------------------------------------------------------------')

    image_batch = build_tfrecord_input(conf, training=True)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    # from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video
    deltat = []
    end = time.time()
    for i_run in range(10000):

        # images, actions, endeff, gen_images, gen_endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos'], dict['gen_images'], dict['gen_states']])
        # images, actions, endeff = sess.run([dict['gen_images'], dict['actions'], dict['endeffector_pos']])
        # images, actions, endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos']])
        [images] = sess.run([image_batch])
        print("read", i_run)

        # file_path = '/'.join(str.split(DATA_DIR, '/')[:-1]+['preview'])
        # comp_single_video(file_path, images, num_exp=conf['batch_size'])
        #
        # # file_path = '/'.join(str.split(DATA_DIR, '/')[:-1] + ['preview_gen_images'])
        # # comp_single_video(file_path, gen_images, num_exp=conf['batch_size'])
        #
        # deltat.append(time.time() - end)
        # end = time.time()

        # show some frames
        # for b in range(conf['batch_size']):

            # print 'actions'
            # print actions[b]
            #
            # print 'endeff'
            # print endeff[b]
            #
            # print 'gen_endeff'
            # print gen_endeff[b]

            # print 'gen_endeff'
            # print gen_endeff[b]

            # print 'video mean brightness', np.mean(images[b])
            # if np.mean(images[b]) < 0.25:
            #     print b
            #     plt.imshow(images[b,0])
            #     plt.show()

            # plt.imshow(images[0, 0])
            # plt.show()
            #
            # pdb.set_trace()

            # print 'robot_pos'
            # print robot_pos
            #
            # print 'object_pos'
            # print object_pos

            # visualize_annotation(conf, images[b], robot_pos[b], object_pos[b])

if __name__ == '__main__':
    main()
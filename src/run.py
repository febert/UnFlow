import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from e2eflow.core.train import Trainer
from e2eflow.experiment import Experiment
from e2eflow.util import convert_input_strings

from e2eflow.cartgripper.sel_images import sel_images

# from e2eflow.kitti.input import KITTIInput
# from e2eflow.kitti.data import KITTIData
from e2eflow.chairs.data import ChairsData
from e2eflow.chairs.input import ChairsInput
# from e2eflow.sintel.data import SintelData
# from e2eflow.sintel.input import SintelInput
# from e2eflow.synthia.data import SynthiaData
# from e2eflow.cityscapes.data import CityscapesData

tf.app.flags.DEFINE_string('ex', 'default',
                           'Name of the experiment.'
                           'If the experiment folder already exists in the log dir, '
                           'training will be continued from the latest checkpoint.')
tf.app.flags.DEFINE_boolean('debug', False,
                            'Enable image summaries and disable checkpoint writing for debugging.')
tf.app.flags.DEFINE_boolean('ow', False,
                            'Overwrites a previous experiment with the same name (if present)'
                            'instead of attempting to continue from its latest checkpoint.')
FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    experiment = Experiment(
        name=FLAGS.ex,
        overwrite=FLAGS.ow)
    dirs = experiment.config['dirs']
    run_config = experiment.config['run']

    gpu_list_param = run_config['gpu_list']

    if isinstance(gpu_list_param, int):
        gpu_list = [gpu_list_param]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list_param)
    else:
        gpu_list = list(range(len(gpu_list_param.split(','))))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_param
    gpu_batch_size = int(run_config['batch_size'] / max(len(gpu_list), 1))
    devices = ['/gpu:' + str(gpu_num) for gpu_num in gpu_list]
    from tensorflow.python.client import device_lib
    print('using device ', device_lib.list_local_devices())

    train_dataset = run_config.get('dataset', 'kitti')

    # kdata = KITTIData(data_dir=dirs['data'],
    #                   fast_dir=dirs.get('fast'),
    #                   stat_log_dir=None,
    #                   development=run_config['development'])
    # einput = KITTIInput(data=kdata,
    #                     batch_size=1,
    #                     normalize=False,
    #                     dims=(384, 1280))

    if train_dataset == 'chairs':
        cconfig = copy.deepcopy(experiment.config['train'])
        cconfig.update(experiment.config['train_chairs'])
        convert_input_strings(cconfig, dirs)
        citers = cconfig.get('num_iters', 0)
        cdata = ChairsData(data_dir=dirs['data'],
                           fast_dir=dirs.get('fast'),
                           stat_log_dir=None,
                           development=run_config['development'])
        cinput = ChairsInput(data=cdata,
                 batch_size=gpu_batch_size,
                 normalize=False,
                 dims=(cconfig['height'], cconfig['width']))
        tr = Trainer(
              lambda shift: cinput.input_raw(swap_images=False,
                                             shift=shift * run_config['batch_size']),
              lambda: einput.input_train_2012(),
              params=cconfig,
              normalization=cinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, citers)

    elif train_dataset == 'kitti':
        kconfig = copy.deepcopy(experiment.config['train'])
        kconfig.update(experiment.config['train_kitti'])
        convert_input_strings(kconfig, dirs)
        kiters = kconfig.get('num_iters', 0)
        kinput = KITTIInput(data=kdata,
                            batch_size=gpu_batch_size,
                            normalize=False,
                            skipped_frames=True,
                            dims=(kconfig['height'], kconfig['width']))
        tr = Trainer(
              lambda shift: kinput.input_raw(swap_images=False,
                                             center_crop=True,
                                             shift=shift * run_config['batch_size']),
              lambda: einput.input_train_2012(),
              params=kconfig,
              normalization=kinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, kiters)

    elif train_dataset == 'cityscapes':
        kconfig = copy.deepcopy(experiment.config['train'])
        kconfig.update(experiment.config['train_cityscapes'])
        convert_input_strings(kconfig, dirs)
        kiters = kconfig.get('num_iters', 0)
        cdata = CityscapesData(data_dir=dirs['data'],
                    fast_dir=dirs.get('fast'),
                    stat_log_dir=None,
                    development=run_config['development'])
        kinput = KITTIInput(data=cdata,
                            batch_size=gpu_batch_size,
                            normalize=False,
                            skipped_frames=False,
                            dims=(kconfig['height'], kconfig['width']))
        tr = Trainer(
              lambda shift: kinput.input_raw(swap_images=False,
                                             center_crop=True,
                                             skip=[0, 1],
                                             shift=shift * run_config['batch_size']),
              lambda: einput.input_train_2012(),
              params=kconfig,
              normalization=kinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, kiters)

    elif train_dataset == 'synthia':
        sconfig = copy.deepcopy(experiment.config['train'])
        sconfig.update(experiment.config['train_synthia'])
        convert_input_strings(sconfig, dirs)
        siters = sconfig.get('num_iters', 0)
        sdata = SynthiaData(data_dir=dirs['data'],
                fast_dir=dirs.get('fast'),
                stat_log_dir=None,
                development=run_config['development'])
        sinput = KITTIInput(data=sdata,
                            batch_size=gpu_batch_size,
                            normalize=False,
                            dims=(sconfig['height'], sconfig['width']))
        tr = Trainer(
              lambda shift: sinput.input_raw(swap_images=False,
                                             shift=shift * run_config['batch_size']),
              lambda: einput.input_train_2012(),
              params=sconfig,
              normalization=sinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, siters)

    elif train_dataset == 'kitti_ft':
        ftconfig = copy.deepcopy(experiment.config['train'])
        ftconfig.update(experiment.config['train_kitti_ft'])
        convert_input_strings(ftconfig, dirs)
        ftiters = ftconfig.get('num_iters', 0)
        ftinput = KITTIInput(data=kdata,
                             batch_size=gpu_batch_size,
                             normalize=False,
                             dims=(ftconfig['height'], ftconfig['width']))
        tr = Trainer(
              lambda shift: ftinput.input_train_gt(40),
              lambda: einput.input_train_2015(40),
              supervised=True,
              params=ftconfig,
              normalization=ftinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, ftiters)

    elif train_dataset == 'cartgripper':
        cconfig = copy.deepcopy(experiment.config['train'])
        cconfig.update(experiment.config['cartgripper'])
        convert_input_strings(cconfig, dirs)
        citers = cconfig.get('num_iters', 0)

        from e2eflow.cartgripper.read_tf_records2 import build_tfrecord_input

        conf = {}
        DATA_DIR = os.environ['VMPC_DATA_DIR'] + '/cartgripper_startgoal_large4step/train'
        conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
        conf['skip_frame'] = 1
        conf['train_val_split'] = 0.95
        conf['sequence_length'] = 4  # 48      # 'sequence length, including context frames.'
        conf['batch_size'] = experiment.config['run']['batch_size']
        conf['context_frames'] = 2
        conf['image_only'] = ''
        conf['orig_size'] = [480, 640]
        conf['visualize'] = False

        # global_step_ = tf.placeholder(tf.int32, name="global_step")
        # train_im = sel_images(train_image, global_step_, citers, 4)
        # val_im = sel_images(val_image, global_step_, citers, 4)

        def make_train(iter_offset):
            train_image = build_tfrecord_input(conf, training=True)
            use_size = tf.constant([384, 512])
            im0 = tf.image.resize_images(train_image[:,0], use_size, method=tf.image.ResizeMethod.BILINEAR)
            im1 = tf.image.resize_images(train_image[:,1], use_size, method=tf.image.ResizeMethod.BILINEAR)
            return [im0, im1]
        def make_val(iter_offset):
            val_image = build_tfrecord_input(conf, training=False)
            use_size = tf.constant([384, 512])
            val_image = tf.image.resize_images(val_image, use_size, method=tf.image.ResizeMethod.BILINEAR)
            im0 = tf.image.resize_images(val_image[:, 0], use_size, method=tf.image.ResizeMethod.BILINEAR)
            im1 = tf.image.resize_images(val_image[:, 1], use_size, method=tf.image.ResizeMethod.BILINEAR)
            return [im0, im1]

        tr = Trainer(
            make_train,
            make_val,
            params=cconfig,
            normalization=[np.array([0., 0., 0.], dtype=np.float32), 1.],  #TODO: try with normalizeation
            train_summaries_dir=experiment.train_dir,
            eval_summaries_dir=experiment.eval_dir,
            experiment=FLAGS.ex,
            ckpt_dir=experiment.save_dir,
            debug=FLAGS.debug,
            interactive_plot=run_config.get('interactive_plot'),
            devices=devices)
        tr.run(0, citers)

    else:
      raise ValueError(
          "Invalid dataset. Dataset must be one of "
          "{synthia, kitti, kitti_ft, cityscapes, chairs}")

    if not FLAGS.debug:
        experiment.conclude()


if __name__ == '__main__':
    tf.app.run()

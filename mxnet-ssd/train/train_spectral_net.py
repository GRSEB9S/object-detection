import logging
import sys
import os
import importlib
import mxnet as mx

from dataset.iterator_cvc import DetIterCVC
from dataset.iterator_kaist import DetIterKAIST

from dataset.custom import Custom
from config.config import cfg
from metric import MultiBoxMetric
from initializer import ScaleInitializer

def train_tir_net(network_name, image_set, dataset_path, batch_size, data_shape, mean_rgb, std_rgb, mean_tir, std_tir, resume,
                  finetune, pretrained, epoch, prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate, weight_decay, val_set,
                  lr_refactor_epoch,
                  lr_refactor_ratio,
                  iter_monitor=0,
                  log_file=None, momentum=0.9, dataset='kaist'):

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if log_file:
        print log_file
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # kvstore
    kv = mx.kvstore.create("device")

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # check args
    if isinstance(data_shape, int):
        data_shape = (data_shape, data_shape)
    assert len(data_shape) == 2, "data_shape must be (h, w) tuple or list or int"
    prefix += '_' + str(data_shape[0])


    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # load dataset
    train_imdb = Custom(image_set, dataset_path, shuffle=True, is_train=True)
    val_imdb = Custom(val_set, dataset_path, shuffle=False, is_train=True)

    if dataset in ['kaist', 'custom']:
        # init data iterator
        train_iter = DetIterKAIST(train_imdb,
                                batch_size,
                                data_shape,
                                mean_rgb,
                                std_rgb,
                                mean_tir,
                                std_tir,
                                cfg.TRAIN.RAND_SAMPLERS,
                                cfg.TRAIN.RAND_MIRROR,
                                cfg.TRAIN.EPOCH_SHUFFLE,
                                cfg.TRAIN.RAND_SEED,
                                is_train=True)

        # save per N epoch, avoid saving too frequently
        resize_epoch = int(cfg.TRAIN.RESIZE_EPOCH)

        if resize_epoch > 1:
            batches_per_epoch = ((train_imdb.num_images - 1) / batch_size + 1) * resize_epoch
            train_iter = mx.io.ResizeIter(train_iter, batches_per_epoch)

        train_iter = mx.io.PrefetchingIter(train_iter)

        if val_imdb:
            val_iter = DetIterKAIST(val_imdb,
                               batch_size,
                               data_shape,
                               mean_rgb,
                               std_rgb,
                               mean_tir,
                               std_tir,
                               cfg.VALID.RAND_SAMPLERS,
                               cfg.VALID.RAND_MIRROR,
                               cfg.VALID.EPOCH_SHUFFLE,
                               cfg.VALID.RAND_SEED,
                               is_train=True)
            val_iter = mx.io.PrefetchingIter(val_iter)
        else:
            val_iter = None

    if dataset == 'cvc':
        # init data iterator
        train_iter = DetIterCVC(train_imdb,
                                  batch_size,
                                  data_shape,
                                  mean_rgb,
                                  std_rgb,
                                  mean_tir,
                                  std_tir,
                                  cfg.TRAIN.RAND_SAMPLERS,
                                  cfg.TRAIN.RAND_MIRROR,
                                  cfg.TRAIN.EPOCH_SHUFFLE,
                                  cfg.TRAIN.RAND_SEED,
                                  is_train=True)

        # save per N epoch, avoid saving too frequently
        resize_epoch = int(cfg.TRAIN.RESIZE_EPOCH)

        if resize_epoch > 1:
            batches_per_epoch = ((train_imdb.num_images - 1) / batch_size + 1) * resize_epoch
            train_iter = mx.io.ResizeIter(train_iter, batches_per_epoch)

        train_iter = mx.io.PrefetchingIter(train_iter)

        if val_imdb:
            val_iter = DetIterCVC(val_imdb,
                                    batch_size,
                                    data_shape,
                                    mean_rgb,
                                    std_rgb,
                                    mean_tir,
                                    std_tir,
                                    cfg.VALID.RAND_SAMPLERS,
                                    cfg.VALID.RAND_MIRROR,
                                    cfg.VALID.EPOCH_SHUFFLE,
                                    cfg.VALID.RAND_SEED,
                                    is_train=True)
            val_iter = mx.io.PrefetchingIter(val_iter)
        else:
            val_iter = None

    # load symbol
    sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    net = importlib.import_module('get_symbol').get_symbol_train(train_imdb.num_classes, network_name)

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # define layers with fixed weight/bias
    fixed_param_names = [name for name in net.list_arguments()
                         if name.startswith('bn0_') or
                         name.startswith('bn_data') or
                         name.startswith('conv0_')]

    for name in net.list_arguments():
        if 'loc' in name.split('_'):
            continue
        if 'pred' in name.split('_'):
            continue
        if name.split('_')[0] in ['stage1',
                               'stage2',
                               'stage3',
                               'stage4', 'conv8', 'conv9', 'conv10', 'pool10', 'bn', 'conv0']:
            fixed_param_names.append(name)

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'


    logger.info("Start training with {} from pretrained model {}".format(ctx_str, pretrained))
    _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)


    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # init training module
    mod = mx.mod.Module(net,
                        data_names=('rgb', 'tir'),
                        label_names=('label',),
                        logger=logger,
                        context=ctx,
                        fixed_param_names=fixed_param_names)

    # fit
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    iter_refactor = lr_refactor_epoch * train_imdb.num_images // train_iter.batch_size
    lr_scheduler = mx.lr_scheduler.FactorScheduler(iter_refactor, lr_refactor_ratio)
    initializer = mx.init.Mixed([".*scale", ".*"], [ScaleInitializer(), mx.init.Xavier(magnitude=1)])

    optimizer_params={'learning_rate':learning_rate,
                      'wd':weight_decay,
                      'lr_scheduler':lr_scheduler,
                      # 'momentum': momentum,
                      'clip_gradient':None,
                      'rescale_grad': 1.0}

    monitor = mx.mon.Monitor(iter_monitor, pattern=".*") if iter_monitor > 0 else None

    mod.fit(train_iter,
            eval_data=val_iter,
            eval_metric=MultiBoxMetric(),
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            optimizer='adam',
            optimizer_params=optimizer_params,
            kvstore=kv,
            begin_epoch=begin_epoch,
            num_epoch=end_epoch,
            initializer=initializer,
            arg_params=args,
            aux_params=auxs,
            allow_missing=True,
            monitor=monitor)





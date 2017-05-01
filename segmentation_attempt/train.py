import os
import mxnet as mx
import logging
import importlib
from file_iter import FileIter
import numpy as np

def jaccard_ind(label, pred):
    return pred


def main():

    # randomly split validation and train data
    train = list(set([x[:8] for x in os.listdir('data/train')]))
    val = list(set([x[:8] for x in os.listdir('data/val')]))

    # get network symbol
    unet = importlib.import_module('symbol_unet').get_unet_symbol()

    # file iterators
    batch_size = 5
    iter_num = 50

    train_iter = FileIter(root_dir="data/train/", data_list=train, batch_size=batch_size, iter_num=iter_num)
    val_iter = FileIter(root_dir="data/val/", data_list=val, batch_size=batch_size, iter_num=5)

    # init training module
    mod = mx.mod.Module(unet, logger=logger, context=ctx, label_names = ('label',))
    batch_end_callback = mx.callback.log_train_metric(2)
    epoch_end_callback = mx.callback.do_checkpoint('unet')

    optimizer_params = {'learning_rate': 0.0001, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=500, factor=0.9)}
    logging.info('Start training...')

    mod.fit(train_data=train_iter,
            eval_data=val_iter,
            eval_metric=mx.metric.np(jaccard_ind),
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            optimizer='adam',
            optimizer_params=optimizer_params,
            begin_epoch=0,
            num_epoch=100,
            initializer=mx.initializer.MSRAPrelu())

if __name__ == "__main__":

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('current_log.txt')
    logger.addHandler(fh)
    ctx = mx.gpu(0)
    main()
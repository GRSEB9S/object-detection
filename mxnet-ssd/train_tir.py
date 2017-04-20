import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
from train.train_tir_net import train_tir_net
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

    # data set params
    parser.add_argument('--dataset', dest='dataset', help='which dataset to use', default='kaist', type=str)
    parser.add_argument('--image-set', dest='image_set', help='train set, can be trainval or train', default='train', type=str)
    parser.add_argument('--val-image-set', dest='val_image_set', help='validation set, can be val or test', default='test', type=str)
    parser.add_argument('--dataset-path', dest='dataset_path', help='dataset path', default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--network', dest='network', type=str, default='spectral', help='which network to use')

    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1, help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1, help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix', default=os.path.join(os.getcwd(), 'model'), type=str)

    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model', default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix', default=os.path.join(os.getcwd(), 'model'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with', default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training', default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training', default=300, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging', default=10, type=int)

    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300, help='set image shape')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.000001, help='weight decay')
    parser.add_argument('--lr-epoch', dest='lr_refactor_epoch', type=int, default=25, help='refactor learning rate every N epoch')
    parser.add_argument('--lr-ratio', dest='lr_refactor_ratio', type=float, default=0.8, help='ratio to refactor learning rate')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log", help='save training log to file')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0, help='log network parameters every N iters if larger than 0')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.dataset_path = args.dataset_path + '/' + args.dataset
    args.pretrained += '/' + args.network + '/' + args.network
    args.prefix += '/' + args.network + '/training_epochs/ssd'
    args.log_file = os.path.join(os.getcwd(), 'model', args.network, "train-"+str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))+".log")

    print
    print "//////////////////////////////////////////////////////////////////////////"
    print 'Parameters : '
    args_dict = vars(args)
    for key, value in args_dict.items():
        print key, ':', value
    print "//////////////////////////////////////////////////////////////////////////"
    print

    if args.dataset == 'kaist':
        mean_rgb = [89.909961557, 83.8302041534, 74.1431794542]
        std_rgb = [65.1171282799, 62.1827802828, 61.1897309395]
        mean_tir = [42.6318449296]
        std_tir = [27.2190767513]

    if args.dataset == 'cvc':
        mean_rgb = []
        std_rgb = []
        mean_tir = []
        std_tir = []

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    ctx = mx.cpu() if not ctx else ctx

    train_tir_net(args.network,
              args.image_set,
              args.dataset_path,
              args.batch_size,
              args.data_shape,
              mean_rgb,
              std_rgb,
              mean_tir,
              std_tir,
              args.resume,
              args.finetune,
              args.pretrained,
              args.epoch,
              args.prefix,
              ctx,
              args.begin_epoch,
              args.end_epoch,
              args.frequent,
              args.learning_rate,
              args.weight_decay,
              args.val_image_set,
              args.lr_refactor_epoch,
              args.lr_refactor_ratio,
              args.monitor,
              args.log_file,
              args.momentum)
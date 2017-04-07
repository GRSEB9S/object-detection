# import symbols
import mxnet as mx

from symbol_caffenet import caffenet
from symbol_squeezenet import squeezenet
from symbol_resnet import resnet

from common import bn_act_conv_layer, multibox_layer


def get_symbol_train(num_classes=20, network='caffenet'):
    """
    Single-shot multi-box detection
    This is a training network with losses

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background

    Returns:
    ----------
    mx.Symbol
    """

    label = mx.symbol.Variable(name="label")

    if network == 'caffenet':
        input_1, input_2, input_3 = caffenet()
    elif network == 'squeezenet':
        input_1, input_2, input_3 = squeezenet()
    elif network in ['resnet', 'resnet_tir']:
        input_1, input_2, input_3 = resnet()

    # ssd extra layers
    conv8_1, elu8_1 = bn_act_conv_layer(input_3, "8_1", 256, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv8_2, elu8_2 = bn_act_conv_layer(conv8_1, "8_2", 512, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    conv9_1, elu9_1 = bn_act_conv_layer(conv8_2, "9_1", 128, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv9_2, elu9_2 = bn_act_conv_layer(conv9_1, "9_2", 256, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    conv10_1, elu10_1 = bn_act_conv_layer(conv9_2, "10_1", 128, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv10_2, elu10_2 = bn_act_conv_layer(conv10_1, "10_2", 256, kernel=(3, 3), pad=(1, 1), stride=(2, 2))

    # global Pooling
    pool10 = mx.symbol.Pooling(data=conv10_2, pool_type="avg", global_pool=True, kernel=(1, 1), name='pool10')

    # with feature maps: ssd_input
    from_layers = [input_1, input_2, conv8_2, conv9_2, conv10_2, pool10]
    sizes = [[.1], [.2, .276], [.38, .461], [.56, .644], [.74, .825], [.92, 1.01]]

    ratios = [[1, 2, .5],
              [1, 2, .5, 3, 1. / 3],
              [1, 2, .5, 3, 1. / 3],
              [1, 2, .5, 3, 1. / 3],
              [1, 2, .5, 3, 1. / 3],
              [1, 2, .5, 3, 1. / 3]]

    normalizations = [20, -1, -1, -1, -1, -1]

    if network in ['resnet', 'resnet_tir']:
        num_channels = [128]
    else:
        num_channels = [256]

    if network == 'squeezenet':
        from_layers = from_layers[2:]
        sizes = sizes[2:]
        ratios = ratios[2:]
        normalizations = normalizations[2:]
        num_channels = []

    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, num_classes, sizes=sizes, ratios=ratios, normalization=normalizations,
                                                        num_channels=num_channels, clip=True, interm_layer=0)

    tmp = mx.contrib.symbol.MultiBoxTarget(*[anchor_boxes, label, cls_preds],
                                   overlap_threshold=.5, ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0,
                                   negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2), name="multibox_target")

    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, ignore_label=-1, use_ignore=True,
                                       grad_scale=1., multi_output=True, normalization='valid', name="cls_prob")
    # loc loss
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", data=loc_target_mask*(loc_preds-loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., normalization='valid', name="loc_loss")

    # confidence loss
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label])
    return out


def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=True, network='caffenet'):
    """
    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        threshold of overlap for non-maximum suppression

    Returns:
    ----------
    mx.Symbol
    """

    net = get_symbol_train(num_classes, network=network)

    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds,
                                           mode='channel',
                                           name='cls_prob')

    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], name="detection", nms_threshold=nms_thresh,
                                      force_suppress=force_suppress, variances=(0.1, 0.1, 0.2, 0.2))
    return out

import mxnet as mx
from common import conv_act_layer, conv_act_layer2
from common import multibox_layer
import tools.find_mxnet



def print_inferred_shape(net, name):
    print name
    ar, ou, au = net.infer_shape(rgb=(1, 3, 300, 300), tir=(1, 1, 300, 300))
    print ou


def residual_unit(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=256):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
    act1 = mx.symbol.LeakyReLU(data=bn1, act_type='elu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True,
                               workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    act2 = mx.symbol.LeakyReLU(data=bn2, act_type='elu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True,
                               workspace=workspace, name=name + '_conv2')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    return conv2 + shortcut


def residual_fusion(res_unit_rgb, res_unit_tir, num_filters=64, name='fusion1_unit_1'):
    concat = mx.symbol.Concat(name=name + '_concat', *[res_unit_rgb, res_unit_tir])

    conv1 = mx.sym.Convolution(concat,
                               num_filter=num_filters,
                               kernel=(3, 3),
                               stride=(1, 1), pad=(1, 1),
                               workspace=256,
                               name=name + '_conv')
    return conv1


def resnet():
    filter_list = [64, 64, 128, 256, 512]
    bn_mom = 0.9
    workspace = 256

    rgb = mx.sym.Variable(name='rgb')
    tir = mx.sym.Variable(name='tir')

    # rgb head
    rgb = mx.sym.BatchNorm(rgb, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='rgb_bn_data')
    net_rgb = mx.sym.Convolution(rgb, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                 no_bias=True, name="rgb_conv0", workspace=workspace)
    net_rgb = mx.sym.BatchNorm(net_rgb, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='rgb_bn0')
    net_rgb = mx.symbol.LeakyReLU(net_rgb, act_type='elu', name='rgb_relu0')
    net_rgb = mx.symbol.Pooling(net_rgb, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    # tir head
    tir = mx.sym.BatchNorm(tir, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='tir_bn_data')
    net_tir = mx.sym.Convolution(tir, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                 no_bias=True, name="tir_conv0", workspace=workspace)
    net_tir = mx.sym.BatchNorm(net_tir, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='tir_bn0')
    net_tir = mx.symbol.LeakyReLU(net_tir, act_type='elu', name='tir_relu0')
    net_tir = mx.symbol.Pooling(net_tir, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    # stage 1
    net_rgb = residual_unit(net_rgb, filter_list[1], (1, 1), False, name='rgb_stage1_unit1', workspace=workspace)
    net_tir = residual_unit(net_tir, filter_list[1], (1, 1), False, name='tir_stage1_unit1', workspace=workspace)
    fusion_1_1 = residual_fusion(net_rgb, net_tir, num_filters=64, name='fusion1_1')

    net_rgb = residual_unit(fusion_1_1, filter_list[1], (1, 1), True, name='rgb_stage1_unit2', workspace=workspace)
    net_tir = residual_unit(net_tir, filter_list[1], (1, 1), True, name='tir_stage1_unit2', workspace=workspace)
    fusion_1_2 = residual_fusion(net_rgb, net_tir, num_filters=64, name='fusion1_2')

    # stage 2
    net_rgb = residual_unit(fusion_1_2, filter_list[2], (2, 2), False, name='rgb_stage2_unit1', workspace=workspace)
    net_tir = residual_unit(net_tir, filter_list[2], (2, 2), False, name='tir_stage2_unit1', workspace=workspace)
    fusion_2_1 = residual_fusion(net_rgb, net_tir, num_filters=128, name='fusion2_1')

    net_rgb = residual_unit(fusion_2_1, filter_list[2], (1, 1), True, name='rgb_stage2_unit2', workspace=workspace)
    net_tir = residual_unit(net_tir, filter_list[2], (1, 1), True, name='tir_stage2_unit2', workspace=workspace)
    fusion_2_2 = residual_fusion(net_rgb, net_tir, num_filters=128, name='fusion2_2')

    # stage 3
    net_rgb = residual_unit(fusion_2_2, filter_list[3], (2, 2), False, name='rgb_stage3_unit1', workspace=workspace)
    net_tir = residual_unit(net_tir, filter_list[3], (2, 2), False, name='tir_stage3_unit1', workspace=workspace)
    fusion_3_1 = residual_fusion(net_rgb, net_tir, num_filters=256, name='fusion3_1')

    net_rgb = residual_unit(fusion_3_1, filter_list[3], (1, 1), True, name='rgb_stage3_unit2', workspace=workspace)
    net_tir = residual_unit(net_tir, filter_list[3], (1, 1), True, name='tir_stage3_unit2', workspace=workspace)
    fusion_3_2 = residual_fusion(net_rgb, net_tir, num_filters=256, name='fusion3_2')

    # stage 4
    net_rgb = residual_unit(fusion_3_2, filter_list[4], (2, 2), False, name='rgb_stage4_unit1', workspace=workspace)
    net_tir = residual_unit(net_tir, filter_list[4], (2, 2), False, name='tir_stage4_unit1', workspace=workspace)
    fusion_4_1 = residual_fusion(net_rgb, net_tir, num_filters=512, name='fusion4_1')

    net_rgb = residual_unit(fusion_4_1, filter_list[4], (1, 1), True, name='rgb_stage4_unit2', workspace=workspace)
    net_tir = residual_unit(net_tir, filter_list[4], (1, 1), True, name='tir_stage4_unit2', workspace=workspace)
    fusion_4_2 = residual_fusion(net_rgb, net_tir, num_filters=512, name='fusion4_2')

    bn1 = mx.sym.BatchNorm(fusion_4_2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.symbol.LeakyReLU(bn1, act_type='elu', name='relu1')
    output = relu1
    return output


def get_symbol_train(num_classes=20):
    """
    This is a training network with losses

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background

    Returns:
    ----------
    mx.Symbol
    """
    # get inputs to extra SSD layers
    label = mx.symbol.Variable(name="label")
    out = resnet()  # 10x10
    internals = out.get_internals()
    feature_maps_38 = internals['rgb_stage3_unit1_relu1_output']  # 38x38
    feature_maps_19 = internals['rgb_stage4_unit1_relu1_output']  # 19X19

    # ssd extra layers
    conv8_1, elu8_1 = conv_act_layer2(out, "8_1", 256, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv8_2, elu8_2 = conv_act_layer2(conv8_1, "8_2", 512, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    conv9_1, elu9_1 = conv_act_layer2(conv8_2, "9_1", 128, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv9_2, elu9_2 = conv_act_layer2(conv9_1, "9_2", 256, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    conv10_1, elu10_1 = conv_act_layer2(conv9_2, "10_1", 128, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv10_2, elu10_2 = conv_act_layer2(conv10_1, "10_2", 256, kernel=(3, 3), pad=(1, 1), stride=(2, 2))

    # global Pooling
    pool10 = mx.symbol.Pooling(data=conv10_2, pool_type="avg", global_pool=True, kernel=(1, 1), name='pool10')

    # ssd settings
    from_layers = [feature_maps_38, feature_maps_19, conv8_2, conv9_2, conv10_2, pool10]
    sizes = [[.1], [.2,.276], [.38, .461], [.56, .644], [.74, .825], [.92, 1.01]]
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
    normalizations = [20, -1, -1, -1, -1, -1]
    num_channels = [128]

    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers,
                                                        num_classes,
                                                        sizes=sizes,
                                                        ratios=ratios,
                                                        normalization=normalizations,
                                                        clip=True,
                                                        num_channels=num_channels,
                                                        interm_layer=0)

    tmp = mx.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds],
        overlap_threshold=.5,
        ignore_label=-1,
        negative_mining_ratio=3,
        minimum_negative_samples=0,
        negative_mining_thresh=.5,
        variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")

    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds,
                                       label=cls_target,
                                       ignore_label=-1,
                                       use_ignore=True,
                                       grad_scale=1.,
                                       multi_output=True,
                                       normalization='valid',
                                       name="cls_prob")

    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_",
                                    data=loc_target_mask * (loc_preds - loc_target),
                                    scalar=1.0)

    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.,
                                  normalization='valid',
                                  name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target,
                                   grad_scale=0,
                                   name="cls_label")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label])
    return out

def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=True):
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
    net = get_symbol_train(num_classes)
    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds,
                                           mode='channel',
                                           name='cls_prob')

    out = mx.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes],
                                      name="detection",
                                      nms_threshold=nms_thresh,
                                      force_suppress=force_suppress,
                                      variances=(0.1, 0.1, 0.2, 0.2))
    return out
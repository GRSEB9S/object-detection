import mxnet as mx
from common import bn_act_conv_layer, conv_act_layer


def residual_unit(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=256):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
    act1 = mx.symbol.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    act2 = mx.symbol.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    return conv2 + shortcut


# fusion functions
def cf_unit(res_unit_rgb, res_unit_tir, num_filters=64, name='fusion1_unit_1', mode='conv'):
    if mode == 'conv':
        concat = mx.symbol.Concat(*[res_unit_rgb, res_unit_tir], dim=1)
        conv = mx.sym.Convolution(concat, num_filter=num_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True, workspace=256, name=name + '_conv')
        conv = mx.symbol.Activation(data=conv, act_type='relu')

    elif mode == 'sum':
        conv = mx.symbol.broadcast_add(res_unit_rgb, res_unit_tir, name=name + '_conv')
        conv = mx.symbol.Activation(data=conv, act_type='relu')

    elif mode == 'max':
        conv = mx.symbol.broadcast_maximum(res_unit_rgb, res_unit_tir, name=name + '_conv')
        conv = mx.symbol.Activation(data=conv, act_type='relu')
    return conv


def spectral_net():
    filter_list = [32, 32, 64, 128, 256]

    tir = mx.sym.Variable(name='tir')

    net_tir = mx.sym.Convolution(tir, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv_0')
    net_tir = mx.symbol.Activation(net_tir, act_type='relu', name='relu_0')
    net_tir = mx.symbol.Pooling(net_tir, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool_0')

    net_tir = mx.sym.Convolution(net_tir, num_filter=filter_list[1], kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv_1')
    net_tir = mx.symbol.Activation(net_tir, act_type='relu', name='relu_1')
    net_tir = mx.symbol.Pooling(net_tir, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool1')

    net_tir = mx.sym.Convolution(net_tir, num_filter=filter_list[2], kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv2')
    net_tir = mx.symbol.Activation(net_tir, act_type='relu', name='relu2')
    net_tir = mx.symbol.Pooling(net_tir, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool2')

    net_tir = mx.sym.Convolution(net_tir, num_filter=filter_list[3], kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv3')
    net_tir = mx.symbol.Activation(net_tir, act_type='relu', name='relu3')
    net_tir = mx.symbol.Pooling(net_tir, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool3')

    net_tir = mx.sym.Convolution(net_tir, num_filter=filter_list[4], kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv4')
    net_tir = mx.symbol.Activation(net_tir, act_type='relu', name='relu4')
    net_tir = mx.symbol.Pooling(net_tir, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool4')

    net_tir, relu9_2t = conv_act_layer(net_tir, "f4", 128, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    net_tir, relu10_2t = conv_act_layer(net_tir, "f5", 128, kernel=(3, 3), pad=(1, 1), stride=(2, 2))

    fusion_1 = net_tir.get_internals()["pool2_output"]
    fusion_2 = net_tir.get_internals()["pool3_output"]
    fusion_3 = net_tir.get_internals()["pool4_output"]
    fusion_4 = net_tir.get_internals()["convf4_output"]
    fusion_5 = net_tir.get_internals()["convf5_output"]

    return [fusion_1, fusion_2, fusion_3, fusion_4, fusion_5]


def resnet():
    filter_list = [64, 64, 128, 256, 512]
    bn_mom = 0.9
    workspace = 256

    rgb = mx.sym.Variable(name='rgb')

    # rgb head
    rgb = mx.sym.BatchNorm(rgb, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    net_rgb = mx.sym.Convolution(rgb, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True, name="conv0", workspace=workspace)
    net_rgb = mx.sym.BatchNorm(net_rgb, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    net_rgb = mx.symbol.Activation(net_rgb, act_type='relu', name='relu0')
    net_rgb = mx.symbol.Pooling(net_rgb, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    # stage 1
    net_rgb = residual_unit(net_rgb, filter_list[1], (1, 1), False, name='stage1_unit1', workspace=workspace)
    net_rgb = residual_unit(net_rgb, filter_list[1], (1, 1), True, name='stage1_unit2', workspace=workspace)

    # stage 2
    net_rgb = residual_unit(net_rgb, filter_list[2], (2, 2), False, name='stage2_unit1', workspace=workspace)
    net_rgb = residual_unit(net_rgb, filter_list[2], (1, 1), True, name='stage2_unit2', workspace=workspace)

    # stage 3
    net_rgb = residual_unit(net_rgb, filter_list[3], (2, 2), False, name='stage3_unit1', workspace=workspace)
    net_rgb = residual_unit(net_rgb, filter_list[3], (1, 1), True, name='stage3_unit2', workspace=workspace)

    # stage 4
    net_rgb = residual_unit(net_rgb, filter_list[4], (2, 2), False, name='stage4_unit1', workspace=workspace)
    net_rgb = residual_unit(net_rgb, filter_list[4], (1, 1), True, name='stage4_unit2', workspace=workspace)

    # ssd extra layers
    extra_layers_input = net_rgb.get_internals()["stage4_unit1_relu1_output"]  # 19x19
    conv8_1, relu8_1 = bn_act_conv_layer(extra_layers_input, "8_1", 256, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv8_2, relu8_2 = bn_act_conv_layer(conv8_1, "8_2", 512, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    conv9_1, relu9_1 = bn_act_conv_layer(conv8_2, "9_1", 128, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv9_2, relu9_2 = bn_act_conv_layer(conv9_1, "9_2", 256, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    conv10_1, relu10_1 = bn_act_conv_layer(conv9_2, "10_1", 128, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv10_2, relu10_2 = bn_act_conv_layer(conv10_1, "10_2", 256, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    pool10 = mx.symbol.Pooling(data=conv10_2, pool_type="avg", global_pool=True, kernel=(1, 1), name='pool10')
    net_rgb = pool10

    fusion_1 = net_rgb.get_internals()["stage3_unit1_relu1_output"]
    fusion_2 = net_rgb.get_internals()["stage4_unit1_relu1_output"]
    fusion_3 = conv8_2
    fusion_4 = conv9_2
    fusion_5 = conv10_2

    return [fusion_1, fusion_2, fusion_3, fusion_4, fusion_5, pool10]

def fusion_net():
    rgb_fusion_layers = resnet()
    tir_fusion_layers = spectral_net()
    input_1 = cf_unit(rgb_fusion_layers[0], tir_fusion_layers[0], num_filters=128, name='fusion1')
    input_2 = cf_unit(rgb_fusion_layers[1], tir_fusion_layers[1], num_filters=256, name='fusion2')
    input_3 = cf_unit(rgb_fusion_layers[2], tir_fusion_layers[2], num_filters=512, name='fusion3')
    input_4 = cf_unit(rgb_fusion_layers[3], tir_fusion_layers[3], num_filters=256, name='fusion4')
    input_5 = cf_unit(rgb_fusion_layers[4], tir_fusion_layers[4], num_filters=256, name='fusion5')
    input_6 = rgb_fusion_layers[5]

    return [input_1, input_2, input_3, input_4, input_5, input_6]



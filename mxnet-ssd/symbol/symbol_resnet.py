import mxnet as mx

def residual_unit(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=256):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
    act1 = mx.symbol.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    act2 = mx.symbol.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True, workspace=workspace, name=name + '_sc')
    return conv2 + shortcut


def resnet():
    filter_list = [64, 64, 128, 256, 512]
    bn_mom = 0.9
    workspace = 256

    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    net = mx.sym.Convolution(data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True, name="conv0", workspace=workspace)
    net = mx.sym.BatchNorm(net, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    net = mx.symbol.Activation(net, act_type='relu', name='relu0')
    net = mx.symbol.Pooling(net, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    # stage 1
    net = residual_unit(net, filter_list[1], (1, 1), False, name='stage1_unit1', workspace=workspace)
    net = residual_unit(net, filter_list[1], (1, 1), True, name='stage1_unit2', workspace=workspace)
    # stage 2
    net = residual_unit(net, filter_list[2], (2, 2), False, name='stage2_unit1', workspace=workspace)
    net = residual_unit(net, filter_list[2], (1, 1), True, name='stage2_unit2', workspace=workspace)
    # stage 3
    net = residual_unit(net, filter_list[3], (2, 2), False, name='stage3_unit1', workspace=workspace)
    net = residual_unit(net, filter_list[3], (1, 1), True, name='stage3_unit2', workspace=workspace)
    # stage 4
    net = residual_unit(net, filter_list[4], (2, 2), False, name='stage4_unit1', workspace=workspace)
    net = residual_unit(net, filter_list[4], (1, 1), True, name='stage4_unit2', workspace=workspace)

    internals = net.get_internals()
    input_1 = internals['stage3_unit1_relu1_output']  # 38x38
    input_2 = internals['stage4_unit1_relu1_output']  # 19X19

    return [input_1, input_2, input_2]


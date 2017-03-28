import mxnet as mx


def squeezenet():
    """
    Squeezenet
    Returns:
    ----------
    specific layers inputs for SSD-Layers
    """

    data = mx.symbol.Variable(name='data')

    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(0, 0), kernel=(3, 3), stride=(2, 2), no_bias=False)
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1, pad=(0, 0), kernel=(3, 3), stride=(2, 2), pool_type='max')
    relu_conv1 = mx.symbol.Activation(name='relu_conv1', data=pool1, act_type='relu')

    fire2_squeeze1x1 = mx.symbol.Convolution(name='fire2_squeeze1x1', data=relu_conv1, num_filter=16, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire2_relu_squeeze1x1 = mx.symbol.Activation(name='fire2_relu_squeeze1x1', data=fire2_squeeze1x1, act_type='relu')
    fire2_expand1x1 = mx.symbol.Convolution(name='fire2_expand1x1', data=fire2_relu_squeeze1x1, num_filter=64, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire2_relu_expand1x1 = mx.symbol.Activation(name='fire2_relu_expand1x1', data=fire2_expand1x1, act_type='relu')
    fire2_expand3x3 = mx.symbol.Convolution(name='fire2_expand3x3', data=fire2_relu_squeeze1x1, num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire2_relu_expand3x3 = mx.symbol.Activation(name='fire2_relu_expand3x3', data=fire2_expand3x3, act_type='relu')
    fire2_concat = mx.symbol.Concat(name='fire2_concat', *[fire2_relu_expand1x1, fire2_relu_expand3x3])

    fire3_squeeze1x1 = mx.symbol.Convolution(name='fire3_squeeze1x1', data=fire2_concat, num_filter=16, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire3_relu_squeeze1x1 = mx.symbol.Activation(name='fire3_relu_squeeze1x1', data=fire3_squeeze1x1, act_type='relu')
    fire3_expand1x1 = mx.symbol.Convolution(name='fire3_expand1x1', data=fire3_relu_squeeze1x1, num_filter=64,pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire3_relu_expand1x1 = mx.symbol.Activation(name='fire3_relu_expand1x1', data=fire3_expand1x1, act_type='relu')
    fire3_expand3x3 = mx.symbol.Convolution(name='fire3_expand3x3', data=fire3_relu_squeeze1x1, num_filter=64,pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire3_relu_expand3x3 = mx.symbol.Activation(name='fire3_relu_expand3x3', data=fire3_expand3x3, act_type='relu')
    fire3_concat = mx.symbol.Concat(name='fire3_concat', *[fire3_relu_expand1x1, fire3_relu_expand3x3])
    pool3 = mx.symbol.Pooling(name='pool3', data=fire3_concat, pad=(0, 0), kernel=(3, 3), stride=(2, 2), pool_type='max')

    fire4_squeeze1x1 = mx.symbol.Convolution(name='fire4_squeeze1x1', data=pool3, num_filter=32, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire4_relu_squeeze1x1 = mx.symbol.Activation(name='fire4_relu_squeeze1x1', data=fire4_squeeze1x1, act_type='relu')
    fire4_expand1x1 = mx.symbol.Convolution(name='fire4_expand1x1', data=fire4_relu_squeeze1x1, num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire4_relu_expand1x1 = mx.symbol.Activation(name='fire4_relu_expand1x1', data=fire4_expand1x1, act_type='relu')
    fire4_expand3x3 = mx.symbol.Convolution(name='fire4_expand3x3', data=fire4_relu_squeeze1x1, num_filter=128, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire4_relu_expand3x3 = mx.symbol.Activation(name='fire4_relu_expand3x3', data=fire4_expand3x3, act_type='relu')
    fire4_concat = mx.symbol.Concat(name='fire4_concat', *[fire4_relu_expand1x1, fire4_relu_expand3x3])

    fire5_squeeze1x1 = mx.symbol.Convolution(name='fire5_squeeze1x1', data=fire4_concat, num_filter=32, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire5_relu_squeeze1x1 = mx.symbol.Activation(name='fire5_relu_squeeze1x1', data=fire5_squeeze1x1, act_type='relu')
    fire5_expand1x1 = mx.symbol.Convolution(name='fire5_expand1x1', data=fire5_relu_squeeze1x1, num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire5_relu_expand1x1 = mx.symbol.Activation(name='fire5_relu_expand1x1', data=fire5_expand1x1, act_type='relu')
    fire5_expand3x3 = mx.symbol.Convolution(name='fire5_expand3x3', data=fire5_relu_squeeze1x1, num_filter=128,pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire5_relu_expand3x3 = mx.symbol.Activation(name='fire5_relu_expand3x3', data=fire5_expand3x3, act_type='relu')
    fire5_concat = mx.symbol.Concat(name='fire5_concat', *[fire5_relu_expand1x1, fire5_relu_expand3x3]) # 36x36
    pool5 = mx.symbol.Pooling(name='pool5', data=fire5_concat, pad=(0, 0), kernel=(3, 3), stride=(2, 2), pool_type='max')

    fire6_squeeze1x1 = mx.symbol.Convolution(name='fire6_squeeze1x1', data=pool5, num_filter=48, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire6_relu_squeeze1x1 = mx.symbol.Activation(name='fire6_relu_squeeze1x1', data=fire6_squeeze1x1, act_type='relu')
    fire6_expand1x1 = mx.symbol.Convolution(name='fire6_expand1x1', data=fire6_relu_squeeze1x1, num_filter=192, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire6_relu_expand1x1 = mx.symbol.Activation(name='fire6_relu_expand1x1', data=fire6_expand1x1, act_type='relu')
    fire6_expand3x3 = mx.symbol.Convolution(name='fire6_expand3x3', data=fire6_relu_squeeze1x1, num_filter=192, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire6_relu_expand3x3 = mx.symbol.Activation(name='fire6_relu_expand3x3', data=fire6_expand3x3, act_type='relu')
    fire6_concat = mx.symbol.Concat(name='fire6_concat', *[fire6_relu_expand1x1, fire6_relu_expand3x3])

    fire7_squeeze1x1 = mx.symbol.Convolution(name='fire7_squeeze1x1', data=fire6_concat, num_filter=48, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire7_relu_squeeze1x1 = mx.symbol.Activation(name='fire7_relu_squeeze1x1', data=fire7_squeeze1x1, act_type='relu')
    fire7_expand1x1 = mx.symbol.Convolution(name='fire7_expand1x1', data=fire7_relu_squeeze1x1, num_filter=192, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire7_relu_expand1x1 = mx.symbol.Activation(name='fire7_relu_expand1x1', data=fire7_expand1x1, act_type='relu')
    fire7_expand3x3 = mx.symbol.Convolution(name='fire7_expand3x3', data=fire7_relu_squeeze1x1, num_filter=192, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire7_relu_expand3x3 = mx.symbol.Activation(name='fire7_relu_expand3x3', data=fire7_expand3x3, act_type='relu')
    fire7_concat = mx.symbol.Concat(name='fire7_concat', *[fire7_relu_expand1x1, fire7_relu_expand3x3])

    fire8_squeeze1x1 = mx.symbol.Convolution(name='fire8_squeeze1x1', data=fire7_concat, num_filter=64, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire8_relu_squeeze1x1 = mx.symbol.Activation(name='fire8_relu_squeeze1x1', data=fire8_squeeze1x1, act_type='relu')
    fire8_expand1x1 = mx.symbol.Convolution(name='fire8_expand1x1', data=fire8_relu_squeeze1x1, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire8_relu_expand1x1 = mx.symbol.Activation(name='fire8_relu_expand1x1', data=fire8_expand1x1, act_type='relu')
    fire8_expand3x3 = mx.symbol.Convolution(name='fire8_expand3x3', data=fire8_relu_squeeze1x1, num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire8_relu_expand3x3 = mx.symbol.Activation(name='fire8_relu_expand3x3', data=fire8_expand3x3, act_type='relu')
    fire8_concat = mx.symbol.Concat(name='fire8_concat', *[fire8_relu_expand1x1, fire8_relu_expand3x3])

    fire9_squeeze1x1 = mx.symbol.Convolution(name='fire9_squeeze1x1', data=fire8_concat, num_filter=64, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire9_relu_squeeze1x1 = mx.symbol.Activation(name='fire9_relu_squeeze1x1', data=fire9_squeeze1x1, act_type='relu')
    fire9_expand1x1 = mx.symbol.Convolution(name='fire9_expand1x1', data=fire9_relu_squeeze1x1, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire9_relu_expand1x1 = mx.symbol.Activation(name='fire9_relu_expand1x1', data=fire9_expand1x1, act_type='relu')
    fire9_expand3x3 = mx.symbol.Convolution(name='fire9_expand3x3', data=fire9_relu_squeeze1x1, num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire9_relu_expand3x3 = mx.symbol.Activation(name='fire9_relu_expand3x3', data=fire9_expand3x3, act_type='relu')
    fire9_concat = mx.symbol.Concat(name='fire9_concat', *[fire9_relu_expand1x1, fire9_relu_expand3x3]) #17x17

    return [None, None, fire9_concat]



import mxnet as mx


def caffenet():
    """
    Fully Convolutional Caffenet
    Returns:
    ----------
    specific layers inputs for SSD-Layers
    """

    data = mx.symbol.Variable(name="data")

    # group 1
    conv1 = mx.symbol.Convolution(data=data, kernel=(11, 11), stride=(4, 4), num_filter=96, name="conv1", num_group=1)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool1")
    relu1 = mx.symbol.LeakyReLU(data=pool1, act_type="prelu", name="prelu1", slope=0.1)

    # group 2
    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), pad=(2, 2), num_filter=256, name="conv2", num_group=2)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool2") # 36x36
    relu2 = mx.symbol.LeakyReLU(data=pool2, act_type="prelu", name="prelu2", slope=0.1)

    # group 3
    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3, 3), pad=(1, 1), num_filter=384, name="conv3", num_group=1)
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3", slope=0.1)

    # group 4
    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384, name="conv4", num_group=2)
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name="prelu4", slope=0.1)

    # group 5
    conv5 = mx.symbol.Convolution(data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv5", num_group=2) # 17X17
    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name="prelu5", slope=0.1)
    pool5 = mx.symbol.Pooling(data=relu5, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool5") # 8x8

    return [relu2, relu5, pool5]





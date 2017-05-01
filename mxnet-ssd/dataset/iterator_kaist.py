from __future__ import division
import mxnet as mx
import numpy as np
import cv2
from tools.rand_sampler import RandSampler

class DetIterKAIST(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch

    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    data_shape : int or (int, int)
        image shape to be resized
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    rand_samplers : list
        random cropping sampler list, if not specified, will
        use original image only
    rand_mirror : bool
        whether to randomly mirror input images, default False
    shuffle : bool
        whether to shuffle initial image list, default False
    rand_seed : int or None
        whether to use fixed random seed, default None
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """
    def __init__(self, imdb,
                 batch_size,
                 data_shape,
                 mean_rgb,
                 std_rgb,
                 mean_tir,
                 std_tir,
                 rand_samplers=[],
                 rand_mirror=False,
                 shuffle=False,
                 rand_seed=None,
                 is_train=True,
                 max_crop_trial=50):
        super(DetIterKAIST, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(data_shape, int):
            data_shape = (data_shape, data_shape)
        self._data_shape = data_shape
        self._mean_tir_pixel = mx.nd.array(mean_tir)
        self._mean_rgb_pixels = mx.nd.array(mean_rgb).reshape((3,1,1))
        self._std_tir_pixel = mx.nd.array(std_tir)
        self._std_rgb_pixels = mx.nd.array(std_rgb).reshape((3,1,1))

        if not rand_samplers:
            self._rand_samplers = []
        else:
            if not isinstance(rand_samplers, list):
                rand_samplers = [rand_samplers]
            assert isinstance(rand_samplers[0], RandSampler), "Invalid rand sampler"
            self._rand_samplers = rand_samplers
        self.is_train = is_train
        self._rand_mirror = rand_mirror
        self._shuffle = shuffle
        if rand_seed:
            np.random.seed(rand_seed) # fix random seed
        self._max_crop_trial = max_crop_trial

        self._current = 0
        self._size = imdb.num_images
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        if self.is_train:
            return [(k, v.shape) for k, v in self._label.items()]
        else:
            return []

    def reset(self):
        self._current = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(), label=self._label.values(), pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current // self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data_rgb = mx.nd.zeros((self.batch_size, 3, self._data_shape[0], self._data_shape[1]))
        batch_data_tir = mx.nd.zeros((self.batch_size, 1, self._data_shape[0], self._data_shape[1]))
        batch_label = []

        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size // 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]


            im_path = self._imdb.image_path_from_index(index)

            with open('/home/home/PycharmProjects/thesis-ssd/mxnet-ssd/data/kaist/images_rgb/' + im_path.split('/')[-1][:-4] + '.jpg', 'rb') as fp:
                img_content = fp.read()
            img_rgb = mx.img.imdecode(img_content)

            with open('/home/home/PycharmProjects/thesis-ssd/mxnet-ssd/data/kaist/images_tir/' + im_path.split('/')[-1][:-4] + '.jpg', 'rb') as fp:
                img_content = fp.read()
            img_tir = mx.img.imdecode(img_content, to_rgb=0, flag=0)
            # img = np.load(im_path)
            # img_rgb = mx.nd.array(img[:,:,:3])
            # img_tir = mx.nd.expand_dims(mx.nd.array(img[:,:,3]), axis=2)
            gt = self._imdb.label_from_index(index).copy() if self.is_train else None
            rgb, tir, label = self._data_augmentation(img_rgb, img_tir, gt)
            batch_data_rgb[i] = rgb
            batch_data_tir[i] = tir
            if self.is_train:
                batch_label.append(label)

        self._data = {'rgb': batch_data_rgb, 'tir': batch_data_tir}

        if self.is_train:
            self._label = {'label': mx.nd.array(np.array(batch_label))}
        else:
            self._label = {'label': None}

    def _data_augmentation(self, rgb, tir, label):
        """
        perform data augmentations: crop, mirror, resize, sub mean, swap channels...
        """
        if self.is_train and self._rand_samplers:
            rand_crops = []
            for rs in self._rand_samplers:
                rand_crops += rs.sample(label)
            num_rand_crops = len(rand_crops)
            # randomly pick up one as input data
            if num_rand_crops > 0:
                index = int(np.random.uniform(0, 1) * num_rand_crops)
                width = rgb.shape[1]
                height = rgb.shape[0]
                crop = rand_crops[index][0]
                xmin = int(crop[0] * width)
                ymin = int(crop[1] * height)
                xmax = int(crop[2] * width)
                ymax = int(crop[3] * height)
                if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:
                    rgb = mx.img.fixed_crop(rgb, xmin, ymin, xmax-xmin, ymax-ymin)
                    tir = mx.img.fixed_crop(tir, xmin, ymin, xmax - xmin, ymax - ymin)
                else:
                    # padding mode
                    new_width = xmax - xmin
                    new_height = ymax - ymin
                    offset_x = 0 - xmin
                    offset_y = 0 - ymin

                    data_bak = rgb
                    rgb = mx.nd.full((new_height, new_width, 3), 128, dtype='uint8')
                    np_data = rgb.asnumpy()
                    np_data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak.asnumpy()
                    rgb = mx.nd.array(np_data)

                    data_bak = tir
                    tir = mx.nd.full((new_height, new_width, 1), 128, dtype='uint8')
                    np_data = tir.asnumpy()
                    np_data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak.asnumpy()
                    tir = mx.nd.array(np_data)

                label = rand_crops[index][1]

        if self.is_train:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]

        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        rgb = mx.img.imresize(rgb, self._data_shape[1], self._data_shape[0], interp_method)
        tir = mx.img.imresize(tir, self._data_shape[1], self._data_shape[0], interp_method)

        if self.is_train and self._rand_mirror:
            if np.random.uniform(0, 1) > 0.5:
                rgb = mx.nd.flip(rgb, axis=1)
                tir = mx.nd.flip(tir, axis=1)
                valid_mask = np.where(label[:, 0] > -1)[0]
                tmp = 1.0 - label[valid_mask, 1]
                label[valid_mask, 1] = 1.0 - label[valid_mask, 3]
                label[valid_mask, 3] = tmp

        rgb = mx.nd.transpose(rgb, (2,0,1))
        rgb = rgb.astype('float32')
        rgb = rgb - self._mean_rgb_pixels

        tir = mx.nd.transpose(tir, (2,0,1))
        tir = tir.astype('float32')
        tir = tir - self._mean_tir_pixel

        return rgb, tir, label

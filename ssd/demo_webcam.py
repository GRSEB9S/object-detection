import mxnet as mx
import numpy as np
import cv2
import time
import argparse
import os
import random

# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow','diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network video demo')
    parser.add_argument('--save-video', dest='save_video', help='the path of the video to save', default=False)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        choices=['vgg16_reduced', 'caffenet', 'squeezenet', 'resnet'], help='which network to use')
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix', default=os.path.join(os.getcwd(), 'model'), type=str)
    parser.add_argument('--epoch', help='epoch num of trained model', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect', action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0, help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300, help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123, help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117, help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104, help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.6, help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5, help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True, help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True, help='show detection time')
    args = parser.parse_args()
    return args


def visualize_detection_img(img, dets, spend_time, color_cls=None, classes=[], thresh=0.6):
    height = img.shape[0]
    width = img.shape[1]

    # draw FPS
    cv2.putText(img=img,
                text='{0:.2f} FPS'.format(1 / spend_time),
                org=(int(20), int(20)),
                fontFace=1,
                fontScale=2,
                color=(255, 255, 255),
                thickness=2)

    for i in range(dets.shape[0]):

        cls_id = int(dets[i, 0])

        if cls_id >= 0:
            score = dets[i, 1]

            if score > thresh:

                if cls_id not in color_cls:
                    color_cls[cls_id] = (random.randrange(0, 255),
                                      random.randrange(0, 255),
                                      random.randrange(0, 255))

                xmin = int(dets[i, 2] * width)
                ymin = int(dets[i, 3] * height)
                xmax = int(dets[i, 4] * width)
                ymax = int(dets[i, 5] * height)

                cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color=color_cls[cls_id], thickness=2)
                class_name = str(cls_id)

                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]

                cv2.putText(img, '{:s} {:.3f}'.format(class_name, score), (xmin, ymin - 2), 0, 0.4, color_cls[cls_id], 1)
    return img


if __name__ == '__main__':

    args = parse_args()
    args.prefix += '/' + args.network + '/deployed/deploy_ssd_300'

    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    predictor = mx.model.FeedForward.load(args.prefix, args.epoch, ctx=ctx, numpy_batch_size=1)

    # generate colors for all classes
    color_cls = dict()
    for i in range(20):
        color_cls[i] = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))

    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    img_shape = img.shape

    if args.save_video:
        # Define the codec and create VideoWriter object/
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 30.0, (img_shape[1], img_shape[0]))


    while(True):
        # Capture frame-by-frame
        ret, img = cap.read()

        if ret:
            data_shape = (args.data_shape, args.data_shape)

            img2 = img.copy()
            img = cv2.resize(img, data_shape, interpolation=cv2.INTER_LINEAR)
            img_arr = np.asarray(img)
            img_arr = img_arr.copy()
            img_arr[:, :, (0, 1, 2)] = img_arr[:, :, (2, 1, 0)]
            img_arr = img_arr.astype(float)
            pixel_means = [args.mean_r, args.mean_g, args.mean_b]
            img_arr -= pixel_means
            channel_swap = (2, 0, 1)
            im_tensor = img_arr.transpose(channel_swap)
            im_tensor = im_tensor[np.newaxis, :]

            start = time.time()
            detections = predictor.predict(im_tensor)
            spend_time = time.time() - start
            det = detections[0, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result = visualize_detection_img(img2, res, spend_time, color_cls, CLASSES, thresh=0.45)

            # Display the resulting frame
            cv2.imshow('result', result)
            if args.save_video:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    if args.save_video:
        out.release()
    cv2.destroyAllWindows()
import numpy as np
import tensorflow as tf
import cv2
import os
# import open3d
import argparse
import fnmatch


def scale_and_crop(im, h, w, K):
    im_h = im.shape[0]
    im_w = im.shape[1]
    scale = w / im_w
    K_cur = K.copy()

    im = cv2.resize(im, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    K_cur[0, 0] *= scale
    K_cur[1, 1] *= scale

    im_h = im.shape[0]
    im_w = im.shape[1]

    off_x = (im_w - w) // 2
    off_y = (im_h - h) // 2
    im = im[off_y: off_y + h, off_x: off_x + w]
    K_cur[0, 2] -= off_x
    K_cur[1, 2] -= off_y

    return im, K_cur


def reverse_scale_and_crop(im, h, w):
    im_h = im.shape[0]
    im_w = im.shape[1]
    im_c = im.shape[2]
    scale = w / im_w

    im = cv2.resize(im, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if im_c == 1:
        im = np.expand_dims(im, axis=-1)
    im_h = im.shape[0]
    im_w = im.shape[1]

    off_x = (w - im_w) // 2
    off_y = (h - im_h) // 2
    ret_im = np.zeros([h, w, im_c], dtype=im.dtype)
    ret_im[off_y: off_y + im_h, off_x: off_x + im_w, :] = im

    return ret_im


def main():
    parser = argparse.ArgumentParser(description='Estimate depth from stereo images')
    parser.add_argument('inputdir',
                        help='input directory')
    parser.add_argument('outputdir',
                        help='output directory')
    parser.add_argument('-v', '--verbose', action="store_true", default=False,
                        help='verbose output')
    parser.add_argument('-g', '--visualize', action="store_true", default=False,
                        help='visualize output')

    args = parser.parse_args()

    # depthLim = 65535
    depthLim = 20000

    left_list = sorted(fnmatch.filter(os.listdir(os.path.join(args.inputdir, 'left_rect')), '*.jpg'))
    right_list = sorted(fnmatch.filter(os.listdir(os.path.join(args.inputdir, 'right_rect')), '*.jpg'))

    if args.verbose:
        print('Number of left images: %d' % len(left_list))
        print('Number of right images: %d' % len(right_list))

    # calibration parameters
    camera_file = os.path.join(args.inputdir, 'camera.txt')
    cam_params = np.fromfile(camera_file, dtype=np.float, sep=' ')
    K = np.eye(3, dtype=np.float)
    # fx
    K[0, 0] = cam_params[0]
    # fy
    K[1, 1] = cam_params[1]
    # cx
    K[0, 2] = cam_params[2]
    # cy
    K[1, 2] = cam_params[3]

    # im_width = cam_params[4]
    # im_height = cam_params[5]

    baseline = cam_params[6]

    # path to .meta
    # model_path = 'data/model-inference-513x257-0'
    # network_im_width = 513
    # network_im_height = 257
    model_path = 'data/model-inference-1025x321-0'
    network_im_width = 1025
    network_im_height = 321
    loader = tf.train.import_meta_graph(model_path + '.meta')

    # filename as input
    input_img_1_tensor = tf.get_default_graph().get_tensor_by_name('Dataloader/read_image/read_png_image/DecodePng:0')
    input_img_2_tensor = tf.get_default_graph().get_tensor_by_name('Dataloader/read_image_1/read_png_image/DecodePng:0')
    disp_left = tf.get_default_graph().get_tensor_by_name("disparities/ExpandDims:0")

    config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=2, intra_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # restore model parameters
        loader.restore(sess, model_path)

        # for graph inspection in tensorboard
        # train_writer = tf.summary.FileWriter('summary', sess.graph)
        # train_writer.flush()

        # print out all variables
        # names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        #for name in names:
        #    print(name)

        for i, entry in enumerate(left_list):
            print('leftIm: ', left_list[i])
            print('rightIm: ', right_list[i])

            left_im = cv2.imread(os.path.join(args.inputdir, 'left_rect', left_list[i]))
            right_im = cv2.imread(os.path.join(args.inputdir, 'right_rect', right_list[i]))

            [left_im_sc, K_sc] = scale_and_crop(left_im, network_im_height, network_im_width, K)
            [right_im_sc, K_sc] = scale_and_crop(right_im, network_im_height, network_im_width, K)

            # run
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_options = tf.RunOptions()
            run_metadata = tf.RunMetadata()
            merged = tf.summary.merge_all()
            summary, disp = sess.run([merged, disp_left],
                                     feed_dict={input_img_1_tensor: left_im_sc,
                                                input_img_2_tensor: right_im_sc},
                                     options=run_options,
                                     run_metadata=run_metadata)
            # train_writer.add_run_metadata(run_metadata, 'run%d' % i, i)
            # train_writer.add_summary(summary, i)
            # train_writer.flush()

            print('output', disp.shape)

            # select a slice from first dimension
            disp = disp[0]
            disp = reverse_scale_and_crop(disp, left_im.shape[0], left_im.shape[1])
            disp = np.maximum(disp, 1.0e-4)

            print('min disp = ', np.min(disp))
            print('max disp = ', np.max(disp))

            fx = K[0, 0]
            # depth in [mm]
            depth = baseline * fx / (disp * network_im_width)
            depthUint = np.uint16(1000 * depth)
            dispLim = 1000 * baseline * fx / (depthLim * network_im_width)
            print('dispLim ', dispLim)
            depth[disp < dispLim] = 0
            depthUint[disp < dispLim] = 0

            # save depth image
            cv2.imwrite(os.path.join(args.outputdir, left_list[i].replace('.jpg', '_depth.png')), depthUint)

            if args.visualize:
                # display image
                cv2.imshow("left", left_im)
                cv2.imshow("right", right_im)
                cv2.imshow("depth", depthUint)
                cv2.waitKey(100)

                points3d = cv2.rgbd.depthTo3d(depth, K)

                # visualization of point cloud for testing purposes
                # pointcloud = open3d.geometry.PointCloud()
                # pointcloud.points = open3d.Vector3dVector(np.reshape(points3d, [-1, 3]))
                # pointcloud.colors = open3d.Vector3dVector(np.reshape(left_im.astype(np.float32)/255.0, [-1, 3]))
                # open3d.draw_geometries([pointcloud])

                cv2.waitKey(-1)


if __name__ == "__main__":
    main()

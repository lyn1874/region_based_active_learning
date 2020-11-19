# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:42:15 2018
This file is for selecting the most uncertain regions in each acquisition step based on different
acquisition functions given the updated model
@author: s161488
"""
import tensorflow as tf
from data_utils.prepare_data import padding_training_data, prepare_train_data
from models.inference import ResNet_V2_DMNN
from models.acquisition_region import select_most_uncertain_patch
import numpy as np
import os


def selection(test_data_statistics_dir, ckpt_dir, acqu_method, already_selected_imindex,
              already_selected_binarymask, kernel_window, stride_size, num_most_uncert_patch, data_path,
              check_overconfident=False):
    """Now the selection method has changed to be region-specific selection and most certain images selection
    test_data_statistics_dir: the test_data_statistics_dir for saving data
    ckpt_dir: already trained model
    acqu_method: ["B"]
    already_selected_imindex: the image index that have been already selected
    already_selected_binarymask: the binary mask that defines the regions that have already been selected
    in the previous acquisition steps
    kernel_window: the size of the anchor
    stride_size: the stride while searching for the most uncertain regions
    num_most_uncert_patch: the maximum number of patches that are selected in per step
    data_path: the directory that saves the pool dataset, default '/home/blia/Exp_Data/Data/glanddata.npy
    check_overconfident: bool variable, it's set to be True when I use this function for analyzing whether the selected
    regions are indeed uncertain
    """
    # --------Here lots of parameters need to be set------Or maybe we could set it in the configuration file-----#
    if check_overconfident is True:
        if not os.path.exists(test_data_statistics_dir):
            os.makedirs(test_data_statistics_dir)
    batch_size = 1
    targ_height_npy = 528  # this is for padding images
    targ_width_npy = 784  # this is for padding images
    ckpt_dir = ckpt_dir
    image_c = 3
    MOVING_AVERAGE_DECAY = 0.999
    num_sample = 1
    num_sample_drop = 30

    Dropout_State = True
    selec_training_index = np.zeros([2, 5])
    selec_training_index[0, :] = [0, 1, 2, 3, 4]  # this is the index for the initial benign images
    selec_training_index[1, :] = [2, 4, 5, 6, 7]  # this is the index for the initial malignant images
    selec_training_index = selec_training_index.astype('int64')

    with tf.Graph().as_default():
        # The placeholder below is for extracting the input for the network #####
        images_train = tf.placeholder(tf.float32, [batch_size, targ_height_npy, targ_width_npy, image_c])
        instance_labels_train = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy, 1])
        edges_labels_train = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy, 1])
        phase_train = tf.placeholder(tf.bool, shape=None, name="training_state")
        dropout_phase = tf.placeholder(tf.bool, shape=None, name="dropout_state")
        data_train, data_pool, data_val = prepare_train_data(data_path, selec_training_index[0, :],
                                                             selec_training_index[1, :])
        x_image_pl, y_label_pl, y_edge_pl = padding_training_data(data_pool[0], data_pool[1], data_pool[2],
                                                                  targ_height_npy, targ_width_npy)

        print("The pooling data size %d" % np.shape(x_image_pl)[0])
        #  Here is for build up the network-----------------------------------------------------------------###
        fb_logits, ed_logits = ResNet_V2_DMNN(images=images_train, training_state=phase_train,
                                              dropout_state=dropout_phase, Num_Classes=2)
        edge_prob = tf.nn.softmax(tf.add_n(ed_logits))
        fb_prob = tf.nn.softmax(tf.add_n(fb_logits))

        var_train = tf.trainable_variables()
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_averages.apply(var_train)
        variables_to_restore = variable_averages.variables_to_restore(tf.moving_average_variables())
        saver = tf.train.Saver(variables_to_restore)

        print(" =====================================================")
        print("Dropout Phase", Dropout_State)
        print("The acquire method", acqu_method)
        print("The number of repeat times", num_sample)
        print("The number of dropout times", num_sample_drop)
        print("The images which include most uncertain patches", already_selected_imindex)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("restore parameter from ", ckpt.model_checkpoint_path)

            for Repeat in range(num_sample):
                ed_prob_tot = []
                fb_prob_tot = []
                fb_prob_var_tot = []
                ed_prob_var_tot = []
                fb_bald_mean_tot = []
                ed_bald_mean_tot = []

                num_image = np.shape(x_image_pl)[0]
                for single_image in range(num_image):
                    feed_dict_op = {images_train: np.expand_dims(x_image_pl[single_image], 0),
                                    instance_labels_train: np.expand_dims(y_label_pl[single_image], 0),
                                    edges_labels_train: np.expand_dims(y_edge_pl[single_image], 0),
                                    phase_train: False,
                                    dropout_phase: Dropout_State}
                    fb_prob_per_image = []
                    ed_prob_per_image = []
                    fb_bald_per_image = []
                    ed_bald_per_image = []
                    fetches_pool = [fb_prob, edge_prob]
                    for single_sample in range(num_sample_drop):
                        _fb_prob, _ed_prob = sess.run(fetches=fetches_pool, feed_dict=feed_dict_op)
                        single_fb_bald = _fb_prob * np.log(_fb_prob + 1e-08)
                        # single_ed_bald = _ed_prob*np.log(_ed_prob+1e-08)
                        fb_bald_per_image.append(single_fb_bald)
                        # ed_bald_per_image.append(single_ed_bald)
                        fb_prob_per_image.append(_fb_prob[0])
                        ed_prob_per_image.append(_ed_prob[0])

                    fb_pred = np.mean(fb_prob_per_image, axis=0)
                    ed_pred = np.mean(ed_prob_per_image, axis=0)

                    fb_prob_tot.append(fb_pred)
                    ed_prob_tot.append(ed_pred)
                    fb_prob_var_tot.append(np.std(fb_prob_per_image, axis=0))
                    ed_prob_var_tot.append(np.std(ed_prob_per_image, axis=0))
                    fb_bald_mean_tot.append(np.mean(fb_bald_per_image, axis=0))
                    # ed_bald_mean_tot.append(np.mean(ed_bald_per_image, axis = 0))

                fb_bald_mean_tot = np.squeeze(np.array(fb_bald_mean_tot), axis=1)
                # ed_bald_mean_tot = np.squeeze(np.array(ed_bald_mean_tot), axis = 1)
                print("Using seletion method", acqu_method)
                most_uncertain_data = select_most_uncertain_patch(x_image_pl, y_label_pl,
                                                                  np.array(fb_prob_tot),
                                                                  np.array(ed_prob_tot),
                                                                  fb_bald_mean_tot,
                                                                  kernel_window, stride_size,
                                                                  already_selected_imindex,
                                                                  already_selected_binarymask,
                                                                  num_most_uncert_patch,
                                                                  acqu_method)
                if check_overconfident is True:
                    np.save(os.path.join(test_data_statistics_dir, 'fbprob'), np.array(fb_prob_tot))
                    if acqu_method is "D":
                        np.save(os.path.join(test_data_statistics_dir, 'fbbald'), fb_bald_mean_tot)

    return most_uncertain_data

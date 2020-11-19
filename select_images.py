# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:42:15 2018
This file is for selecting the most uncertain images in each acquisition step based on the updated model
@author: s161488
"""

import tensorflow as tf
from data_utils.prepare_data import padding_training_data, prepare_train_data
from models.inference import ResNet_V2_DMNN
from models.acquistion_full_image import extract_informative_index

import numpy as np


def selection(test_data_statistics_dir, ckpt_dir, acqu_method, acqu_index, num_select_point_from_pool, agg_method,
              agg_quantile_cri, data_path='/home/blia/Exp_Data/Data/glanddata.npy', save=False):
    # --------Here lots of parameters need to be set------Or maybe we could set it in the configuration file-----#
    if save is True:
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
        y_imindex_pl = np.array(data_pool[-2])
        y_clsindex_pl = np.array(data_pool[-1])

        if acqu_index is not None:
            for remove_data_row in range(np.shape(acqu_index)[0]):
                print("Number of benign and malignant samples in previous selection",
                      y_clsindex_pl[acqu_index[remove_data_row, :]])
                removed_image_index = y_imindex_pl[acqu_index[remove_data_row, :]]
                print("Already selected image index", removed_image_index)
                image_tot_index = range(np.shape(x_image_pl)[0])
                image_index_left = np.delete(image_tot_index, acqu_index[remove_data_row, :])
                x_image_pl = x_image_pl[image_index_left]
                y_label_pl = y_label_pl[image_index_left]
                y_edge_pl = y_edge_pl[image_index_left]
                y_clsindex_pl = y_clsindex_pl[image_index_left]
                y_imindex_pl = y_imindex_pl[image_index_left]
                print([a in removed_image_index for a in y_imindex_pl])
                print("The shape of pool data after selection", np.shape(x_image_pl)[0])

        # ------------------------------Here is for build up the network-------------------------------------###
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
        print("The images which needs to removed from pool set are:", acqu_index)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("restore parameter from ", ckpt.model_checkpoint_path)

            ArgIndex = np.zeros([num_select_point_from_pool, np.shape(acqu_method)[0]])
            for Repeat in range(num_sample):
                ed_prob_tot = []
                fb_prob_tot = []
                fb_prob_var_tot = []
#                ed_prob_var_tot = []
                fb_bald_mean_tot = []
#                ed_bald_mean_tot = []

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
                        single_ed_bald = _ed_prob * np.log(_ed_prob + 1e-08)
                        fb_bald_per_image.append(single_fb_bald)
#                        ed_bald_per_image.append(single_ed_bald)
                        fb_prob_per_image.append(_fb_prob[0])
#                        ed_prob_per_image.append(_ed_prob[0])

                    fb_pred = np.mean(fb_prob_per_image, axis=0)
#                    ed_pred = np.mean(ed_prob_per_image, axis=0)

                    fb_prob_tot.append(fb_pred)
#                    ed_prob_tot.append(ed_pred)
                    fb_prob_var_tot.append(np.std(fb_prob_per_image, axis=0))
#                    ed_prob_var_tot.append(np.std(ed_prob_per_image, axis=0))
                    fb_bald_mean_tot.append(np.mean(fb_bald_per_image, axis=0))
#                    ed_bald_mean_tot.append(np.mean(ed_bald_per_image, axis=0))

                fb_bald_mean_tot = np.squeeze(np.array(fb_bald_mean_tot), axis=1)
#                ed_bald_mean_tot = np.squeeze(np.array(ed_bald_mean_tot), axis=1)
                print("Using seletion method", acqu_method)
                acqu_method_index = 0
                for single_acqu_method in acqu_method:
                    ArgIndex[:, acqu_method_index] = extract_informative_index(single_acqu_method, x_image_pl,
                                                                               np.array(fb_prob_tot),
                                                                               np.array(fb_prob_var_tot),
                                                                               fb_bald_mean_tot,
                                                                               num_select_point_from_pool,
                                                                               agg_method, agg_quantile_cri)

                    print("Finish method", single_acqu_method)
                    acqu_method_index = acqu_method_index + 1

            # np.save(os.path.join(test_data_statistics_dir, 'stat_tot'), stat_tot)
            # np.save(os.path.join(test_data_statistics_dir, 'fbprob'), fb_prob_tot)
            # np.save(os.path.join(test_data_statistics_dir, 'edprob'), ed_prob_tot)
            # np.save(os.path.join(test_data_statistics_dir, 'fbprob_var'), fb_prob_var_tot)
            # np.save(os.path.join(test_data_statistics_dir, 'edprob_var'), ed_prob_var_tot)
            # np.save(os.path.join(test_data_statistics_dir, 'fb_bald_mean'), fb_bald_mean_tot)
            # np.save(os.path.join(test_data_statistics_dir, 'ed_bald_mean'), ed_bald_mean_tot)

    return ArgIndex

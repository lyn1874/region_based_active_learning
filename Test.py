# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:42:15 2018
This file is for testing the already trained active learning model
@author: s161488
"""
import tensorflow as tf
from data_utils.prepare_data import padding_training_data, prepare_test_data
from models.inference import ResNet_V2_DMNN
import os
import numpy as np
import scipy.io as sio


print("--------------------------------------------------------------")
print("---------------DEFINE YOUR TRAINING DATA PATH-----------------")
print("--------------------------------------------------------------")
data_home = 'DATA/Data/'
print("--------------------------------------------------------------")
print("---------------DEFINE YOUR TRAINING DATA PATH-----------------")
print("--------------------------------------------------------------")


def running_test_for_all_acquisition_steps(path_input, version_use, start_step, pool_or_test):
    """This function evaluates the experiments for multiple acquisition steps
    Args:
        path_input: "Method_B_Stage_1_Version_0", the parent directory of each acquisition step ckpt dir
        version_use: int
        start_step: int, default 0, where to start the evaluation
        pool_or_test: whether I am evaluating the experiments on the pool dataset or test dataset, string,
            "pool" or "test"

    Note:
        the path_input and version_use argument can be omitted depends on the place that you save your experiments
    """
    data_kind = ['Test_Data_A/', 'Test_Data_B/']
    if pool_or_test is "test":
        data_name = ['glanddata_testa.npy', 'glanddata_testb.npy']
    else:
        data_name = ["glanddata.npy"]
    path_mom = os.path.join('/scratch/blia/Act_Learn_Desperate_V%d/' % version_use, path_input)
    path_all_cousin = np.load(os.path.join(path_mom, 'total_select_folder.npy'))[start_step:]
    if pool_or_test is "test":
        test_data_init_path = os.path.join(path_mom, 'Test_Data')
    else:
        test_data_init_path = os.path.join(path_mom, 'Pool_Data')
    for path_single_cousin in path_all_cousin:
        ckpt_to_read = path_mom + path_single_cousin.strip().split(path_input)[-1]
        #        ckpt_to_read = path_single_cousin
        for data_path_single, data_name_single in zip(data_kind, data_name):
            single_folder_name = path_single_cousin.strip().split('/')[-2]
            test_data_statistics_dir = os.path.join(test_data_init_path, single_folder_name)
            test_data_statistics_dir = os.path.join(test_data_statistics_dir, data_path_single)
            if not os.path.exists(test_data_statistics_dir):
                os.makedirs(test_data_statistics_dir)
            test_data_path = os.path.join(data_home, data_name_single)
            test(test_data_statistics_dir, ckpt_to_read, test_data_path, save_stat=True)
#            save_image_to_mat(test_data_statistics_dir, test_data_path)


def running_test_for_single_acquisition_step(model_dir):
    """This function saves the prediction for a single acquisition step
    It performs the prediction for Group A and Group B images separately
    model_dir: the directory that saves the model ckpt
    Note:
        the data_home needs to be manually determined!
    """
    data_kind = ['Test_Data_A/', 'Test_Data_B/']
    data_name = ['glanddata_testa.npy', 'glanddata_testb.npy']
    tds = model_dir.strip().split('/FE')[0] + "/Test_Data/%s/" % model_dir.strip().split('/')[-2]
    if not os.path.exists(tds):
        os.makedirs(tds)
    print("the test directory----", tds)
    for data_path_single, data_name_single in zip(data_kind, data_name):
        test_data_statistics_dir = os.path.join(tds, data_path_single)
        if not os.path.exists(test_data_statistics_dir):
            os.makedirs(test_data_statistics_dir)
        test_data_path = os.path.join(data_home, data_name_single)
        test(test_data_statistics_dir, model_dir, test_data_path, save_stat=True)
#        save_image_to_mat(test_data_statistics_dir, test_data_path)


def test(test_data_statistics_dir, ckpt_dir, test_data_path, save_stat=True):
    """This function evaluates the performance of the model
    Args:
        test_data_statistics_dir: the tds_dir that is used for saving data
        ckpt_dir: the directory that saves the model ckpt
        test_data_path: the path that saves the test data. because there are gland A and gland B
        save_stat: bool, whether save the fb probability and ed probability, default to be True
    """
    # --------Here lots of parameters need to be set------Or maybe we could set it in the configuration file-----#
    batch_size = 1
    targ_height_npy = 528  # this is for padding images
    targ_width_npy = 784  # this is for padding images
    ckpt_dir = ckpt_dir
    image_c = 3
    MOVING_AVERAGE_DECAY = 0.999
    check_effect_of_dropout_iter = False
    #  if check_effect_of_dropout_iter is True, then the goal is to find a good number of dropout times which can
    #  achieve good segmentation accuracy
    if check_effect_of_dropout_iter is True:
        sample_size_space = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    else:
        sample_size_space = [30]
    with tf.Graph().as_default():
        # The placeholder below is for extracting the input for the network #####
        images_train = tf.placeholder(tf.float32, [batch_size, targ_height_npy, targ_width_npy, image_c])
        instance_labels_train = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy])
        edges_labels_train = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy])
        phase_train = tf.placeholder(tf.bool, shape=None, name="training_state")
        dropout_phase = tf.placeholder(tf.bool, shape=None, name="dropout_state")
        # -----------------Here is for preparing the dataset for training, pooling and validation---------###
        x_val_group = prepare_test_data(test_data_path)
        x_image_val, y_label_val, y_edge_val = padding_training_data(x_val_group[0], x_val_group[1], x_val_group[2],
                                                                     targ_height_npy,
                                                                     targ_width_npy)
        print("The real validation data is actually the test data from group B %d" % np.shape(x_image_val)[0])

        # ------------------------------Here is for build up the network-----------------------------###
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
        with tf.Session() as sess:
            ckpt_all = tf.train.get_checkpoint_state(ckpt_dir)
            ckpt_use = ckpt_dir + '/' + ckpt_all.model_checkpoint_path.strip().split('/')[-1]
            if os.path.isfile(ckpt_use + '.index'):
                saver.restore(sess, ckpt_use)
                print("restore parameter from", ckpt_use)
            ed_prob_tot = []
            fb_prob_tot = []
            fb_bald_tot = []
            num_image = np.shape(x_image_val)[0]

            print("fb loss, fb acc, ed loss, ed accu")
            tot_stat = np.zeros([num_image, 2])
            for single_image in range(num_image):
                feed_dict_op = {images_train: np.expand_dims(x_image_val[single_image], 0),
                                instance_labels_train: np.expand_dims(y_label_val[single_image], 0),
                                edges_labels_train: np.expand_dims(y_edge_val[single_image], 0),
                                phase_train: False,
                                dropout_phase: True}
                fb_prob_per_image_tot = []
                ed_prob_per_image_tot = []

                fb_bald_per_image_tot = []

                for sample_iter, single_sample_size in enumerate(sample_size_space):
                    fb_prob_per_image = []
                    ed_prob_per_image = []

                    fb_bald_per_image = []

                    for single_sample in range(single_sample_size):
                        _fb_prob, _ed_prob = sess.run([fb_prob, edge_prob], feed_dict=feed_dict_op)
                        fb_prob_per_image.append(_fb_prob)
                        ed_prob_per_image.append(_ed_prob)

                        single_fb_bald = _fb_prob * np.log(_fb_prob + 1e-08)
                        # single_ed_bald = _ed_prob*np.log(_ed_prob+1e-08)
                        fb_bald_per_image.append(single_fb_bald)

                    fb_prob_per_im_avg = np.mean(fb_prob_per_image, axis=0)
                    ed_prob_per_im_avg = np.mean(ed_prob_per_image, axis=0)
                    fb_bald_per_im_avg = np.mean(fb_bald_per_image, axis=0)

                    fb_prob_per_image_tot.append(fb_prob_per_im_avg)
                    ed_prob_per_image_tot.append(ed_prob_per_im_avg)
                    fb_bald_per_image_tot.append(fb_bald_per_im_avg)

                if check_effect_of_dropout_iter is False:
                    fb_prob_tot.append(fb_prob_per_image_tot)
                    ed_prob_tot.append(ed_prob_per_image_tot)
                    fb_bald_tot.append(fb_bald_per_image_tot)
                    # fb_prob_var_tot.append(fb_prob_per_image_var_tot)
                    # ed_prob_var_tot.append(ed_prob_per_image_var_tot)

            if save_stat is False:
                pass
            else:
                np.save(os.path.join(test_data_statistics_dir, 'fbprob'), fb_prob_tot)
                np.save(os.path.join(test_data_statistics_dir, 'edprob'), ed_prob_tot)
                np.save(os.path.join(test_data_statistics_dir, 'fbbald'), fb_bald_tot)


def save_image_to_mat(test_data_statistics_dir, test_data_path):
    """"This function saves the npy stat into mat stat in order to evaluate the segmentation accuracy in Matlab"""
    targ_height_npy = 528  # this is for padding images
    targ_width_npy = 784  # this is for padding images
    x_im_group = prepare_test_data(test_data_path)
    x_image_val, y_label_val, y_edge_val = padding_training_data(x_im_group[0], x_im_group[1], x_im_group[2],
                                                                 targ_height_npy, targ_width_npy)
    fb_prob = np.squeeze(np.squeeze(np.load(os.path.join(test_data_statistics_dir, 'fbprob.npy')), 1), 1)
    ed_prob = np.squeeze(np.squeeze(np.load(os.path.join(test_data_statistics_dir, 'edprob.npy')), 1), 1)

    mask_final_a = []
    ed_prob_cut = []
    fb_prob_cut = []
    for index in range(np.shape(ed_prob)[0]):
        sele_index = np.where(np.mean(x_image_val[index], -1) != 0)
        ed_prob_single = ed_prob[index]
        fb_prob_single = fb_prob[index]
        mask_gt_single = y_label_val[index][np.min(sele_index[0]):np.max(sele_index[0] + 1),
                         np.min(sele_index[1]):np.max(sele_index[1] + 1)]
        ed_prob_cut.append(ed_prob_single[np.min(sele_index[0]):np.max(sele_index[0] + 1),
                           np.min(sele_index[1]):np.max(sele_index[1] + 1), :])
        fb_prob_cut.append(fb_prob_single[np.min(sele_index[0]):np.max(sele_index[0] + 1),
                           np.min(sele_index[1]):np.max(sele_index[1] + 1), :])
        mask_final_a.append(mask_gt_single != 0)
    sio.savemat(os.path.join(test_data_statistics_dir, 'edge_pred.mat'), {'pred': ed_prob_cut})
    sio.savemat(os.path.join(test_data_statistics_dir, 'fb_pred.mat'), {'pred': fb_prob_cut})
    sio.savemat(os.path.join(test_data_statistics_dir, 'mask.mat'), {'ground_truth': mask_final_a})
    os.remove(os.path.join(test_data_statistics_dir, 'fbprob.npy'))
    os.remove(os.path.join(test_data_statistics_dir, 'edprob.npy'))

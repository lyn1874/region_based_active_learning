#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:57:19 2018
This file contains all the function for preparing data in the region-specific active learning scenario
@author: s161488
"""
import numpy as np
from data_utils.prepare_data import prepare_train_data, padding_training_data


def give_init_train_and_val_data(mom_data_path):
    """This function is used to prepare the most initial data
    The shape of the training data will be [10,528,784,3]
    The shape of the val data will be [10,528,784,3]
    The training data will be expanded with the selected images and patches from each acquisition step 
    The validation data will always be same for all the acquisition step
    """
    print("-----------------------Now I am reading the initial training data, which should include 10 images")
#    mom_data_path = "/home/s161488/Exp_Stat/Data/glanddata.npy"
    targ_height_npy = 528  # this is for padding images
    targ_width_npy = 784  # this is for padding images
    selec_training_index = np.zeros([2, 5])
    selec_training_index[0, :] = [0, 1, 2, 3, 4]  # this is the index for the initial benign images
    selec_training_index[1, :] = [2, 4, 5, 6, 7]  # this is the index for the initial malignant images
    selec_training_index = selec_training_index.astype('int64')
    data_train, _, data_val = prepare_train_data(mom_data_path, selec_training_index[0, :],
                                                 selec_training_index[1, :])
    x_image_tr, y_label_tr, y_edge_tr = padding_training_data(data_train[0], data_train[1], data_train[2],
                                                              targ_height_npy, targ_width_npy)
    x_image_val, y_label_val, y_edge_val = padding_training_data(data_val[0], data_val[1], data_val[2],
                                                                 targ_height_npy, targ_width_npy)
    y_label_tr = (y_label_tr != 0).astype('int64')
    y_label_val = (y_label_val != 0).astype('int64')

    y_binarymask_tr = np.ones(np.shape(y_label_tr))
    y_binarymask_val = np.ones(np.shape(y_label_val))

    most_init_tr_data = [x_image_tr, y_label_tr, y_edge_tr, y_binarymask_tr, data_train[-1]]
    all_time_val_data = [x_image_val, y_label_val, y_edge_val, y_binarymask_val]
    return most_init_tr_data, all_time_val_data


def update_training_data(most_init_train_data, most_certain_data, most_uncert_data):
    update_group = []
    for orig, new in zip(most_init_train_data, most_uncert_data):
        update_group.append(np.concatenate([orig, new], axis=0))
    return update_group


# def prepare_the_new_certain_input(Most_Certain_Before, Most_Certain_After):
#     """this function is used to generate the new most certain input.
#     the shape of the most_certain_before and most_certain_After should be both 5,.....
#     simply concatenate the before and after value"""
#     X_im_b, Y_la_b, Y_ed_la_b, Y_bi_b = Most_Certain_Before
#     X_im_a, Y_la_a, Y_ed_la_a, Y_bi_a = Most_Certain_After
#     Total_X_Update = np.concatenate([X_im_b, X_im_a], axis=0)
#     Total_Y_La_Update = np.concatenate([Y_la_b, Y_la_a], axis=0)
#     Total_Y_Ed_La_Update = np.concatenate([Y_ed_la_b, Y_ed_la_a], axis=0)
#     Total_Y_Bi_Update = np.concatenate([Y_bi_b, Y_bi_a], axis=0)
#     Updated_Most_Certain = [Total_X_Update, Total_Y_La_Update, Total_Y_Ed_La_Update, Total_Y_Bi_Update]
#     return Updated_Most_Certain


def prepare_the_new_uncertain_input(most_uncertain_before, most_uncertain_after):
    most_uncertain_before = [np.array(v) for v in most_uncertain_before]
    most_uncertain_after = [np.array(v) for v in most_uncertain_after]
    y_imindex_b, y_imindex_a = most_uncertain_before[-1], most_uncertain_after[-1]
    common_index = [v for v in y_imindex_b if v in y_imindex_a]
    if not common_index:
        total_update_group = []
        for i in range(5):
            total_update_group.append(np.concatenate([most_uncertain_before[i],
                                                      most_uncertain_after[i]], axis=0))
    else:
        print("there are common images between previous acquisition step and current acquisition step")
        print("the common image index are", common_index)
        old_select_index = [i for i, v in enumerate(y_imindex_b) if v in common_index]
        new_select_index = [i for i, v in enumerate(y_imindex_a) if v in common_index]
        old_unique_index = np.delete(np.arange(np.shape(y_imindex_b)[0]), old_select_index)
        new_unique_index = np.delete(np.arange(np.shape(y_imindex_a)[0]), new_select_index)

        print("there are %d unique images in the previous acquisition step" % np.shape(old_unique_index)[0])
        print("there are %d unique images in the current acquisition step" % np.shape(new_unique_index)[0])

        # [print(np.shape(v)) for v in most_uncertain_before]
        # [print(np.shape(v)) for v in most_uncertain_after]
        # print(old_unique_index)
        # print(new_unique_index)

        if len(old_unique_index) > 0:
            old_unique_group = []
            for i in range(5):
                old_unique_group.append(most_uncertain_before[i][old_unique_index])
        if len(new_unique_index) > 0:
            new_unique_group = []
            for i in range(5):
                new_unique_group.append(most_uncertain_after[i][new_unique_index])

        common_stat_group = [[] for i in range(5)]
        for single_ind in common_index:
            old_ind = [i for i, v in enumerate(y_imindex_b) if v == single_ind][0]
            new_ind = [i for i, v in enumerate(y_imindex_a) if v == single_ind][0]
            print("if this value is zero, then it means the images in previous and current step are same",
                  np.sum(most_uncertain_before[0][old_ind] - most_uncertain_after[0][new_ind]))
            for stat_ind in range(5):
                if stat_ind == 0:
                    common_stat_group[stat_ind].append(most_uncertain_after[stat_ind][new_ind])
                elif stat_ind == 1 or stat_ind == 2:
                    _stat_before = most_uncertain_before[stat_ind][old_ind] * most_uncertain_before[-2][old_ind]
                    _stat_after = most_uncertain_after[stat_ind][new_ind]  # * most_uncertain_after[-2][new_ind]
                    _la_after = 1 - most_uncertain_before[-2][old_ind]
                    _stat_aggre = ((_stat_before + _stat_after * _la_after) != 0).astype('int64')
                    common_stat_group[stat_ind].append(_stat_aggre)
                elif stat_ind == 3:
                    _stat_update = ((most_uncertain_before[stat_ind][old_ind] +
                                    most_uncertain_after[stat_ind][new_ind]) != 0).astype('int64')
                    common_stat_group[stat_ind].append(_stat_update)
                elif stat_ind == 4:
                    common_stat_group[stat_ind].append(single_ind)

        total_update_group = []
        # [print(np.shape(v)) for v in common_stat_group]

        if len(old_unique_index) > 0:
            if len(new_unique_index) == 0:
                for i in range(5):
                    total_update_group.append(np.concatenate([old_unique_group[i], np.array(common_stat_group[i])],
                                                             axis=0))
            else:
                for i in range(5):
                    total_update_group.append(np.concatenate([old_unique_group[i], np.array(common_stat_group[i]),
                                                              new_unique_group[i]], axis=0))
        else:
            if len(new_unique_index) == 0:
                for i in range(5):
                    total_update_group.append(common_stat_group[i])
            else:
                for i in range(5):
                    total_update_group.append(np.concatenate([np.array(common_stat_group[i]), new_unique_group[i]],
                                                             axis=0))
    x_update, y_update, y_ed_update, y_bi_update, imind_update = total_update_group
    print("-------shape of the updated data------")
    print([np.shape(v) for v in total_update_group])
    return x_update, y_update, y_ed_update, y_bi_update, imind_update


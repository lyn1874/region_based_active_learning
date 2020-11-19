#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:45:49 2018
This file include all the acquisition function
There is one option for all the functions in this file is that the aggregation method could be different
1. we consider the utility score for all the pixels in per image, therefore, it would be a sum over all the pixels 
2. we consider the most uncertain pixels in per image, therfore it would be like we select the quantile criterior,
 and only consider
the pixels whose utility score is larger than that criterior 
3. It's on the way, I don't know it yet.
@author: s161488
"""
import numpy as np


def extract_uncertainty_index(images, fb_prob, agg_method, quantile_cri):
    num_image = np.shape(fb_prob)[0]
    uncert = np.zeros([num_image, 1])
    for i in range(num_image):
        sele_index = np.where(np.mean(images[i, :, :, :], -1) != 0)
        fb_prob_single = fb_prob[i, np.min(sele_index[0]):np.max(sele_index[0] + 1),
                         np.min(sele_index[1]):np.max(sele_index[1] + 1), :]
        fb_index = np.argmax(fb_prob_single, axis=-1)
        fb_prob_map = 1 - (fb_index * fb_prob_single[:, :, 1] + (1 - fb_index) * fb_prob_single[:, :, 0])
        fb_prob_reshape = np.reshape(fb_prob_map, [-1])
        if agg_method == 'Simple_Sum':
            uncert[i, 0] = np.sum(fb_prob_reshape)
        elif agg_method == 'Quantile':
            num_quant = np.percentile(fb_prob_reshape, q=quantile_cri)
            uncert[i, 0] = np.sum(fb_prob_reshape[fb_prob_reshape >= num_quant])
        else:
            print("Hey, the aggregation method is on its way :)")
    return uncert


def extract_entropy_index(fb_prob, images, agg_method, quantile_cri):
    num_image = np.shape(images)[0]
    entropy_value = np.zeros([num_image, 1])
    for i in range(num_image):
        sele_index = np.where(np.mean(images[i, :, :, :], -1) != 0)
        fb_prob_single = fb_prob[i, np.min(sele_index[0]):np.max(sele_index[0] + 1),
                         np.min(sele_index[1]):np.max(sele_index[1] + 1), :]
        fb_entropy = np.sum(-fb_prob_single * np.log(fb_prob_single + 1e-8),
                            axis=-1)  # calculate the sum w.r.t the number of classes
        fb_entropy_reshape = np.reshape(fb_entropy, [-1])
        if agg_method == 'Simple_Sum':
            entropy_value[i, 0] = np.sum(fb_entropy_reshape)
        elif agg_method == 'Quantile':
            num_quant = np.percentile(fb_entropy_reshape, q=quantile_cri)
            entropy_value[i, 0] = np.sum(fb_entropy_reshape[fb_entropy_reshape >= num_quant])
        else:
            print("Hey, the aggregation method is on its way :)")
    return entropy_value


def extract_bald_index(fb_prob_mean_bald, fb_prob, x_image_pl, agg_method, quantile_cri):
    """This is for acquiring image based on BALD method
    Args:
        fb_prob_mean_bald: shape [Number_of_Image, im_h, im_w, 2]
        fb_prob_mean_bald = 1/t*p_c*log(p_c)
        fb_prob: the predicted probability, shape [Number_of_Image, im_h, im_w, 2]
        x_image_pl: [num_image, imh, imw, 3]
        agg_method: "sum", "quantile"
        quantile_cri: int
    Return:
        BALD_Value
    """
    BALD_value = np.zeros([np.shape(x_image_pl)[0], 1])
    for i in range(np.shape(x_image_pl)[0]):
        sele_index = np.where(np.mean(x_image_pl[i, :, :, :], -1) != 0)
        fb_prob_mean_bald_single = fb_prob_mean_bald[i, np.min(sele_index[0]):np.max(sele_index[0] + 1),
                                   np.min(sele_index[1]):np.max(sele_index[1] + 1), :]
        fb_prob_single = fb_prob[i, np.min(sele_index[0]):np.max(sele_index[0] + 1),
                         np.min(sele_index[1]):np.max(sele_index[1] + 1), :]
        bald_first_term = -np.sum(fb_prob_single * np.log(fb_prob_single + 1e-08), axis=-1)
        bald_second_term = np.sum(fb_prob_mean_bald_single, axis=-1)
        bald_value = bald_first_term + bald_second_term
        bald_reshape = np.reshape(bald_value, [-1])
        if agg_method == 'Simple_Sum':
            BALD_value[i, 0] = np.sum(bald_reshape)
        elif agg_method == 'Quantile':
            num_quant = np.percentile(bald_reshape, q=quantile_cri)
            BALD_value[i, 0] = np.sum(bald_reshape[bald_reshape >= num_quant])
        else:
            print("Hey, the aggregation method is on its way :)")
    return BALD_value


def extract_informative_index(acq_method, x_image_pl, fb_prob, fb_prob_var, fb_prob_mean_bald, num_select_point,
                             agg_method, quantile_cri):
    if acq_method is "B":
        print("acquisition function is uncertainty")
        margin_diff = extract_uncertainty_index(x_image_pl, fb_prob, agg_method, quantile_cri)
    elif acq_method is "C":
        print("acquisition function is entropy")
        margin_diff = extract_entropy_index(fb_prob, x_image_pl, agg_method, quantile_cri)
    elif acq_method is "D":
        print("acquisition function is BALD")
        margin_diff = extract_bald_index(fb_prob_mean_bald, fb_prob, x_image_pl, agg_method, quantile_cri)
    else:
        print("Hey, the acquisition function is on its way :)")
    marg_index = np.argsort(margin_diff[:, 0], axis=0)
    Acq_Index = marg_index[-num_select_point:]
    return Acq_Index


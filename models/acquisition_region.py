#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:03:32 2018
This file is the new version for calculating the uncertainty value in each patch
It's better because:
    1. the way of choosing the most uncertain patch is automate
    2. The weight ratio for each regions can be easily changed to any value
    @author: s161488
"""
import numpy as np
from scipy import signal, ndimage
from skimage.morphology import dilation, disk


def select_most_uncertain_patch(x_image_pl, y_label_pl, fb_pred, ed_pred, fb_prob_mean_bald, kernel_window, stride_size,
                                already_select_image_index, previously_selected_binary_mask, num_most_uncert_patch,
                                method):
    """This function is used to acquire the #most uncertain patches in the pooling set.
    Args:
        x_image_pl: [Num_Im, Im_h, Im_w,3]
        y_label_pl: [Num_Im, Im_h, Im_w,1]
        fb_pred: [Num_Im, Im_h, Im_w, 2]
        ed_pred: [Num_Im, Im_h, Im_w, 2]
        fb_prob_mean_bald: [num_im, imw, imw]
        kernel_window: [kh, kw] determine the size of the region
        stride_size: int, determine the stride between every two regions
        already_select_image_index: if it's None, then it means that's the first acquistion step,
            otherwise it's the numeric image index for the previously selected patches
        previously_selected_binary_mask: [num_already_selected_images, Im_h, Im_w,1]
        num_most_uncert_patch: int, number of patches that are selected in each acquisition step
        method: acquisition method: 'B', 'C', 'D'
    Returns:
        Most_Uncert_Im: [Num_Selected, Im_h, Im_w, 3]imp
        Most_Uncert_FB_GT: [Num_Selected, Im_h, Im_w,1]
        Most_Uncert_ED_GT: [Num_Selected, Im_h, Im_w,1]
        Most_Uncert_Binary_Mask: [Num_Selected, Im_h, Im_w,1]
        Selected_Image_Index: [Num_Selected]
    """
    num_im = np.shape(x_image_pl)[0]
    uncertainty_map_tot = []
    for i in range(num_im):
        if method == 'B':
            var_stat = get_uncert_heatmap(x_image_pl[i], fb_pred[i])
        elif method == 'C':
            var_stat = get_entropy_heatmap(fb_pred[i])
        elif method == 'D':
            var_stat = get_bald_heatmap(fb_prob_mean_bald[i], fb_pred[i])
        uncertainty_map_tot.append(var_stat)
    uncertainty_map_tot = np.array(uncertainty_map_tot)
    if already_select_image_index is None:
        print("--------This is the beginning of the selection process-------")
    else:
        print(
            "----------Some patches have already been annotated, I need to deal with that")
        previously_selected_binary_mask = np.squeeze(previously_selected_binary_mask, axis=-1)
        for i in range(np.shape(previously_selected_binary_mask)[0]):
            uncertainty_map_single = uncertainty_map_tot[already_select_image_index[i]]
            uncertainty_map_updated = uncertainty_map_single * (1 - previously_selected_binary_mask[i])
            uncertainty_map_tot[already_select_image_index[i]] = uncertainty_map_updated
    selected_numeric_image_index, binary_mask_updated_tot = calculate_score_for_patch(uncertainty_map_tot,
                                                                                      kernel_window, stride_size,
                                                                                      num_most_uncert_patch)
    pseudo_fb_la_tot = []
    pseudo_ed_la_tot = []
    for index, single_selected_image_index in enumerate(selected_numeric_image_index):
        pseudo_fb_la, pseudo_ed_la = return_pseudo_label(y_label_pl[single_selected_image_index],
                                                         fb_pred[single_selected_image_index],
                                                         ed_pred[single_selected_image_index],
                                                         binary_mask_updated_tot[index])

        pseudo_fb_la_tot.append(pseudo_fb_la)
        pseudo_ed_la_tot.append(pseudo_ed_la)
    most_uncert_im_tot = x_image_pl[selected_numeric_image_index]
    most_uncertain = [most_uncert_im_tot,
                      pseudo_fb_la_tot,
                      pseudo_ed_la_tot,
                      binary_mask_updated_tot,
                      selected_numeric_image_index]
    return most_uncertain


def calculate_score_for_patch(uncert_est, kernel, stride_size, num_most_uncertain_patch):
    """This function is used to calculate the utility score for each patch.
    Args:
        uncert_est: [num_image, imh, imw]
        kernel: the size of each searching shape
        stride_size: the stride between every two regions
        num_most_uncertain_patch: int, the number of selected regions
    Returns:
        most_uncert_image_index: [Num_Most_Selec] this should be the real image index
        %most_uncert_patch_index: [Num_Most_Selec] this should be the numeric index for the selected patches
        binary_mask: [Num_Most_Selec, Im_h, Im_w,1]
        %pseudo_label: [Num_Most_Selec, Im_h, Im_w,1]
    """
    num_im, imh, imw = np.shape(uncert_est)
    kh, kw = np.shape(kernel)
    h_num_patch = imh - np.shape(kernel)[0] + 1
    w_num_patch = imw - np.shape(kernel)[1] + 1
    num_row_wise = h_num_patch // stride_size
    num_col_wise = w_num_patch // stride_size
    if stride_size == 1:
        tot_num_patch_per_im = num_row_wise * num_col_wise
    else:
        tot_num_patch_per_im = (num_row_wise + 1) * (num_col_wise + 1)
    print('-------------------------------There are %d patches in per image' % tot_num_patch_per_im)
    patch_tot = []
    for i in range(num_im):
        patch_subset = select_patches_in_image_area(uncert_est[i], kernel, stride_size, num_row_wise, num_col_wise)
        patch_tot.append(np.reshape(patch_subset, [-1]))
    patch_tot = np.reshape(np.array(patch_tot), [-1])
    # print('Based on the experiments, there are %d patches in total'%np.shape(patch_tot)[0])
    # print('Based on the calculation, there supposed to be %d patches in tot'%(Num_Im*tot_num_patch_per_im))
    sorted_index = np.argsort(patch_tot)
    select_most_uncert_patch = (sorted_index[-num_most_uncertain_patch:]).astype('int64')
    select_most_uncert_patch_imindex = (select_most_uncert_patch // tot_num_patch_per_im).astype('int64')
    select_most_uncert_patch_index_per_im = (select_most_uncert_patch % tot_num_patch_per_im).astype('int64')
    if stride_size == 1:
        select_most_uncert_patch_rownum_per_im = (select_most_uncert_patch_index_per_im // num_col_wise).astype('int64')
        select_most_uncert_patch_colnum_per_im = (select_most_uncert_patch_index_per_im % num_col_wise).astype('int64')
    else:
        select_most_uncert_patch_rownum_per_im = (select_most_uncert_patch_index_per_im // (num_col_wise + 1)).astype(
            'int64')
        select_most_uncert_patch_colnum_per_im = (select_most_uncert_patch_index_per_im % (num_col_wise + 1)).astype(
            'int64')
    transfered_rownum, transfered_colnum = transfer_strid_rowcol_backto_nostride_rowcol(
        select_most_uncert_patch_rownum_per_im,
        select_most_uncert_patch_colnum_per_im,
        [h_num_patch, w_num_patch],
        [num_row_wise + 1, num_col_wise + 1],
        stride_size)

    binary_mask_tot = []
    # print("The numeric index for the selected most uncertain patches-----", select_most_uncert_patch)
    # print("The corresponding uncertainty value in the selected patch-----", patch_tot[select_most_uncert_patch])
    # print("The image index for the selected most uncertain patches-------", select_most_uncert_patch_imindex)
    # print("The index of the patch in per image---------------------------", select_most_uncert_patch_index_per_im)
    # print("The row index for the selected patch--------------------------", select_most_uncert_patch_rownum_per_im)
    # print("The col index for the selected patch--------------------------", select_most_uncert_patch_colnum_per_im)
    # print("The transfered row index for the selected patch---------------", transfered_rownum)
    # print("The transfered col index for the selected patch---------------", transfered_colnum)

    for i in range(num_most_uncertain_patch):
        single_binary_mask = generate_binary_mask(imh, imw,
                                                  transfered_rownum[i],
                                                  transfered_colnum[i],
                                                  kh, kw)
        binary_mask_tot.append(single_binary_mask)
    binary_mask_tot = np.array(binary_mask_tot)
    unique_im_index = np.unique(select_most_uncert_patch_imindex)
    if np.shape(unique_im_index)[0] == num_most_uncertain_patch:
        print("----------------------------There is no replication for the selected images")
        uncertain_info = [select_most_uncert_patch_imindex, binary_mask_tot]
    else:
        print("-----These images have been selected more than twice", unique_im_index)
        binary_mask_final_tot = []
        for i in unique_im_index:
            loc_im = np.where(select_most_uncert_patch_imindex == i)[0].astype('int64')
            binary_mask_combine = (np.sum(binary_mask_tot[loc_im], axis=0) != 0).astype('int64')
            binary_mask_final_tot.append(binary_mask_combine)
        uncertain_info = [unique_im_index.astype('int64'), np.array(binary_mask_final_tot)]
    print("the shape for binary mask", np.shape(binary_mask_final_tot))
    return uncertain_info


def return_pseudo_label(single_gt, single_fb_pred, single_ed_pred, single_binary_mask):
    """This function is used to return the pseudo label for the selected patches in per image
    Args:
        single_gt: [imh, imw,1]
        single_fb_pred: [imh, imw, 2]
        single_ed_pred: [imh, imw, 2]
        single_binary_mask: [imh, imw]
    Return:
        pseudo_fb_la: [Im_h, Im_w, 1]
        pseudo_ed_la: [Im_h, Im_w, 1]
    """
    single_gt = (single_gt != 0).astype('int64')
    edge_gt = extract_edge(single_gt)
    fake_pred = (single_fb_pred[:, :, -1:] >= 0.5).astype('int64')
    fake_ed_pred = (single_ed_pred[:, :, -1:] >= 0.2).astype('int64')
    print(np.shape(fake_pred), np.shape(single_binary_mask), np.shape(single_gt), np.shape(edge_gt))
    pseudo_fb_la = fake_pred * (1 - single_binary_mask) + single_gt * single_binary_mask
    pseudo_ed_la = fake_ed_pred * (1 - single_binary_mask) + edge_gt * single_binary_mask
    return pseudo_fb_la, pseudo_ed_la


def extract_edge(la_sep):
    """This function is utilized to extract the edge from the ground truth
    Args:
        la_sep [im_h, im_w]
    Return 
        edge_gt [im_h, im_w]
    """
    selem = disk(3)
    sx = ndimage.sobel(la_sep, axis=0, mode='constant')
    sy = ndimage.sobel(la_sep, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    row = (np.reshape(sob, -1) > 0) * 1
    edge_sep = np.reshape(row, [np.shape(sob)[0], np.shape(sob)[1]])
    edge_sep = dilation(edge_sep, selem)
    edge_sep = np.expand_dims(edge_sep, axis=-1)

    return edge_sep.astype('int64')


def generate_binary_mask(imh, imw, rowindex, colindex, kh, kw):
    """This function is used to generate the binary mask for the selected most uncertain images
    Args:
        Im_h, Im_w are the size of the binary mask
        row_index, col_index are the corresponding row and column index for most uncertain patch
        kh,kw are the kernel size
    Output:
        Binary_Mask
    Opts: 
        To transform from the selected patch index to the original image. It will be like
        rowindex:rowindex+kh
        colindex:colindex+kw
    """
    binary_mask = np.zeros([imh, imw, 1])
    binary_mask[rowindex:(rowindex + kh), colindex:(colindex + kw)] = 1
    return binary_mask


def transfer_strid_rowcol_backto_nostride_rowcol(rownum, colnum, no_stride_row_col, stride_row_col, stride_size):
    """This function is used to map the row index and col index from the strided version back to the original version
    if the row_num and col_num are not equal to the last row num or last col num
    then the transfer is just rownum*stride_size, colnum*stride_size
    but if the row_num and colnum are actually the last row num or last col num
    then the transfer is that rownum*stride_size, colnum_no_stride, or row_num_no_stride, colnum*stride_size
    """
    if stride_size != 1:
        row_num_no_stride, col_num_no_stride = no_stride_row_col
        row_num_stride, col_num_stride = stride_row_col
        transfered_row_num = np.zeros([np.shape(rownum)[0]])
        for i in range(np.shape(rownum)[0]):
            if rownum[i] != (row_num_stride - 1):
                transfered_row_num[i] = stride_size * rownum[i]
            else:
                transfered_row_num[i] = row_num_no_stride - 1
        transfered_col_num = np.zeros([np.shape(colnum)[0]])
        for i in range(np.shape(colnum)[0]):
            if colnum[i] != (col_num_stride - 1):
                transfered_col_num[i] = colnum[i] * stride_size
            else:
                transfered_col_num[i] = col_num_no_stride - 1
    else:
        transfered_row_num = rownum
        transfered_col_num = colnum
    return transfered_row_num.astype('int64'), transfered_col_num.astype('int64')


def select_patches_in_image_area(single_fb, kernel, stride_size, num_row_wise, num_col_wise):
    """There needs to be a stride"""
    utility_patches = signal.convolve(single_fb, kernel, mode='valid')
    if stride_size != 1:
        subset_patch = np.zeros([num_row_wise + 1, num_col_wise + 1])
        for i in range(num_row_wise):
            for j in range(num_col_wise):
                subset_patch[i, j] = utility_patches[i * stride_size, j * stride_size]
        for i in range(num_row_wise):
            subset_patch[i, -1] = utility_patches[i * stride_size, -1]
        for j in range(num_col_wise):
            subset_patch[-1, j] = utility_patches[-1, j * stride_size]
        subset_patch[-1, -1] = utility_patches[-1, -1]
    else:
        subset_patch = utility_patches
    return subset_patch


def get_uncert_heatmap(image_single, fb_prob_single, check_rank=False):
    if check_rank is True:
        sele_index = np.where(np.mean(image_single, -1) != 0)
        fb_prob_single = fb_prob_single[np.min(sele_index[0]):np.max(sele_index[0] + 1),
                            np.min(sele_index[1]):np.max(sele_index[1] + 1), :]
    else:
        fb_prob_single = fb_prob_single
    fb_index = (fb_prob_single[:, :, 1] >= 0.5).astype('int64')
    fb_prob_map = fb_index * fb_prob_single[:, :, 1] + (1 - fb_index) * fb_prob_single[:, :, 0]
    only_base_fb = 1 - fb_prob_map
    return only_base_fb


def get_entropy_heatmap(fb_prob_single):
    fb_entropy = np.sum(-fb_prob_single * np.log(fb_prob_single + 1e-8),
                        axis=-1)  # calculate the sum w.r.t the number of classes
    return fb_entropy


def get_bald_heatmap(fb_prob_mean_bald_single, fb_prob_single):
    bald_first_term = -np.sum(fb_prob_single * np.log(fb_prob_single + 1e-08), axis=-1)
    bald_second_term = np.sum(fb_prob_mean_bald_single, axis=-1)
    bald_value = bald_first_term + bald_second_term
    return bald_value

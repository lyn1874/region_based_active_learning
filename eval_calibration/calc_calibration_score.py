# -*- coding: utf-8 -*-
"""
Created on Wed May  20 12:49 2020
This file is for calculating the calibration score of the experiments
@author: s161488
"""
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import shutil

import eval_calibration.calibration_lib as calib
from data_utils.prepare_data import collect_test_data, prepare_pool_data


def get_group_region(method_use):
    """Give the calibration score for region active learning"""
    if method_use is "B":
        path_input = ["Method_B_Stage_1_Version_3", "Method_B_Stage_1_Version_4"]
    elif method_use is "C":
        path_input = ["Method_C_Stage_2_Version_1", "Method_C_Stage_2_Version_2"]
    elif method_use is "D":
        path_input = ["Method_D_Stage_3_Version_2", "Method_D_Stage_3_Version_3"]
    for single_path in path_input:
        calc_calibration_value(single_path, 6)


def get_group_full(version_use):
    """Give the calibration score for full image active learning"""
    path_input = sorted(os.listdir("/scratch/blia/Act_Learn_Desperate_V%d/" % version_use))
    for single_path in path_input:
        calc_calibration_value(single_path, version_use)


def calculate_aggre_pixel(path_input, version, path2read=None):
    """This function calculates the acquired number of pixels in per step
    based on the data in collect
    """
    if not path2read:
        path2read = '/scratch/blia/Act_Learn_Desperate_V%d/%s/' % (version, path_input)
    path2save = '/home/blia/Exp_Data/calibration_score/Act_Learn_Desperate_V%d/' % version
    version = int(path_input.strip().split('_')[-1])
    num_of_pixel_old = np.load(path2read + '/num_of_pixel.npy')
    num_of_image_old = np.load(path2read + '/num_of_image.npy')
    acq_step = len(os.listdir(path2read + '/collect_data/'))
    new_stat = np.zeros([acq_step, 2])
    im_index = []
    unique_im = np.zeros([acq_step])
    for i in range(acq_step):
        p = ["FE_step_00_version_%d" % version if i == 0 else "FE_step_%d_version_%d" % (i - 1, version)][0]
        stat = pickle.load(open(path2read + '/collect_data/%s/updated_uncertain.txt' % p, 'rb'))
        new_stat[i, 0] = np.sum(stat[-2])
        im_index.append(stat[-1])
        _im_ind = np.unique([v for j in im_index for v in j])
        unique_im[i] = len(_im_ind)
    new_stat[:, 0] = np.cumsum(new_stat[:, 0])
    new_stat[:, 1] = unique_im + 10
    print("number of acquired pixels old", num_of_pixel_old)
    print("number of acquired pixels new", new_stat[:, 0])
    print("number of acquired images old", num_of_image_old)
    print("number of acquired images new", new_stat[:, 1])
    np.save(path2save + 'query_stat_%s' % path_input, new_stat)


def transfer_numeric_index_back_to_imindex(numeric_index):
    im_index_pool = np.arange(65)
    numeric_index = numeric_index.astype('int32')
    selected_imindex = np.zeros(np.shape(numeric_index))
    for i in range(len(numeric_index)):
        select_index = numeric_index[i]
        selected_imindex[i] = im_index_pool[select_index]
        im_index_pool = np.delete(im_index_pool, select_index)
    return selected_imindex.astype('int32')


def collect_pool_data_multi_acquisition_step(version, use_str):
    """Args:
    version: the experiment version
    use_str: "Method_B_", or "Method_C_" or "Method_D_"
    """
    path = "/scratch/blia/Act_Learn_Desperate_V%d/" % version
    _, y_label_gt, _ = prepare_pool_data("/home/blia/Exp_Data/Data/glanddata.npy", True)
    path_group = [v for v in os.listdir(path) if use_str in v]
    for single_path in path_group:
        collect_pool_data(version, single_path, y_label_gt)


def collect_pool_data(version, path_use, pool_fb_label):
    """This function returns the predicted uncertainty at each acquisition step for different
    acquisition functions
    Args:
        version: the experiment version
        path_use: the experiment path, such as "Method_B_Stage_1_Version_0"
        pool_fb_label: the pixel-wise ground truth label for images in the pool set
    """
    exp_version = int(path_use.strip().split('_')[-1])
    path = '/scratch/blia/Act_Learn_Desperate_V%d/%s/' % (version, path_use)
    save_dir = '/home/blia/Exp_Data/calibration_score/Act_Learn_Desperate_V%d/' % version
    numeric_index = np.load(path + 'total_acqu_index.npy')
    selected_imindex = transfer_numeric_index_back_to_imindex(numeric_index)
    pool_data_dir = path + 'Pool_Data/'
    fb_group = []
    gt_group = []
    bald_group = []
    print(path_use)
    for i in range(len(numeric_index))[:-3]:
        print("acquisition step ", i)
        fb_ = np.load(pool_data_dir + "/FE_step_%d_version_%d/Test_Data_A/fbprob.npy" % (i, exp_version))
        fb_group.append(fb_[selected_imindex[i + 1]])
        if "_D_" in path_use:
            bald_ = np.load(pool_data_dir + "/FE_step_%d_version_%d/Test_Data_A/fbbald.npy" % (i, exp_version))
            bald_group.append(bald_[selected_imindex[i + 1]])
        gt_group.append(pool_fb_label[selected_imindex[i + 1]])
    if "_D_" in path_use:
        fb_group = [fb_group, bald_group]
    np.save(save_dir + path_use + "pool_label", np.array(gt_group))
    np.save(save_dir + path_use + "pool_stat", np.array(fb_group))


def give_calibration_histogram(path_group, version, test_step, method):
    """This function gives the ece histogram (expected calibration error) at different step
    for different acquisition functions"""
    x_image_test, y_label_test = collect_test_data(resize=False)
    y_label_binary = (y_label_test != 0).astype('int32')
    y_label_binary = np.reshape(y_label_binary, [-1])
    save_dir = "/home/blia/Exp_Data/save_fig/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path_ = "/scratch/blia/Act_Learn_Desperate_V%d/" % version
    stat_group = []
    for single_path in path_group:
        exp_version = int(single_path.strip().split("_")[-1])
        path_use = path_ + single_path + "/Test_Data/"
        _stat = _give_calibration_histogram(path_use, exp_version, test_step, y_label_binary)
        stat_group.append(_stat)
    np.save(save_dir + "/ece_historgram_stat_%d_%s_%d" % (test_step, method, version), stat_group)
    accu_mean = np.mean([v[1] for v in stat_group], axis=0)
    accu_std = np.std([v[1] for v in stat_group], axis=0) * 1.95 / np.sqrt(len(path_group))
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.plot(stat_group[0][0], accu_mean, 'r')
    ax.fill_between(stat_group[0][0], accu_mean - accu_std, accu_mean + accu_std,
                    color='r', alpha=0.5)
    ax.plot([0.0, 1.0], [0.0, 1.0], color='b', ls=':')
    plt.savefig(save_dir + '/%d_%s_%d.jpg' % (test_step, method, version))


def _give_calibration_histogram(path2read, exp_version, test_step, y_label_binary):
    path2read = path2read + "FE_step_%d_version_%d/" % (test_step, exp_version)
    path_a, path_b = "Test_Data_A", "Test_Data_B"
    fb_prob = []
    for single_path in [path_a, path_b]:
        fb = np.load(path2read + single_path + "/fbprob.npy")
        fb = np.reshape(fb, [-1, 2])
        fb_prob.append(fb)
    fb_prob = np.concatenate(fb_prob, axis=0)
    top_k_probs, is_correct = calib.get_multiclass_predictions_and_correctness(fb_prob,
                                                                               y_label_binary, None)
    top_k_probs = top_k_probs.flatten()
    is_correct = is_correct.flatten()
    bin_edges, accuracies, counts = calib.bin_predictions_and_accuracies(top_k_probs,
                                                                         is_correct,
                                                                         bins=10)
    bin_centers = calib.bin_centers_of_mass(top_k_probs, bin_edges)
    return bin_centers, accuracies, counts


def collect_region_uncertainty(version, path_subset, step):
    pathmom = '/scratch/blia/Act_Learn_Desperate_V%d/%s' % (version, path_subset)
    model_version = int(path_subset.strip().split('_')[-1])
    method = str(path_subset.strip().split('_')[1])
    path2read = pathmom + '/collect_data/'
    print("====loading experiment statistics from folder:", path2read)
    for i in range(step):
        print("===step %d=====" % i)
        path = path2read + 'FE_step_%d_version_%d/' % (i, model_version)
        _region_uncertainty(path, method)


def _region_uncertainty(path, method):
    """Calculates the uncertainty from the region acquisition"""
    savepath = '/home/blia/Exp_Data/calibration_score/region_uncertainty/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    path_split = path.strip().split('/')
    savename = path_split[3].strip().split('_')[-1] + '_' + path_split[4] + '_step_' + path_split[6].strip().split('_')[
        2]
    stat = pickle.load(open(path + 'updated_uncertain.txt', 'rb'))
    selected_image, fb_label, ed_label, binary_mask, imindex = stat
    fbprob = np.load(path + 'fbprob.npy')
    fbprob_subset = fbprob[imindex]
    if method is "D":
        fbbald = np.load(path + 'fbbald.npy')
        fbbald_subset = fbbald[imindex]
        fbprob_subset = [fbprob_subset, fbbald_subset]
    uncert = calc_uncertainty(fbprob_subset, method, False)
    print("-----maximum of uncertainty %.2f minimum of uncertainty %.2f------" % (np.max(uncert),
                                                                                  np.min(uncert)))
    uncert = (uncert - np.min(uncert)) / (np.max(uncert) - np.min(uncert))
    uncert = uncert * binary_mask[:, :, :, 0]
    uncert_aggre = uncert[uncert != 0]

    print(np.shape(uncert_aggre), np.sum(binary_mask))
    np.save(savepath + '/' + savename, uncert_aggre)


def calc_uncertainty(prob, method, reshape=True):
    if method is "B":
        uncert = 1 - np.max(prob, axis=-1)
    elif method is "C":
        uncert = np.sum(-prob * np.log(prob + 1e-8), axis=-1)
    elif method is "D":
        prob, bald = prob
        bald_first = -np.sum(prob * np.log(prob + 1e-8), axis=-1)
        bald_second = np.sum(bald, axis=-1)
        uncert = bald_first + bald_second
    if reshape:
        return np.reshape(uncert, [-1])
    else:
        return uncert


def aggregate_stat(version):
    """This function aggregates all the calibration score in one folder
    Args:
        version: int, Act_Learn_Desperate_%d % version
    Ops:
        1. The calibration score needs to be read
        2. Then this calibration score file will copied to a home directory
    """
    path2read = '/scratch/blia/Act_Learn_Desperate_V%d' % version
    path2save = '/home/blia/Exp_Data/calibration_score/'
    path2save = path2save + 'Act_Learn_Desperate_V%d' % version
    if not os.path.exists(path2save):
        os.makedirs(path2save)
    all_model = sorted([v for v in os.listdir(path2read) if 'Method_' in v])
    for single_model in all_model:
        print(single_model)
        orig_file_path = path2read + '/' + single_model + '/' + 'calibration_score.obj'
        new_file_path = path2save + '/%s.obj' % single_model
        shutil.copy(orig_file_path, new_file_path)


def calc_calibration_value(path_input, version_use):
    """The calculated calibration score here is used to create Figure 1, 5 and E2 in the paper"""
    x_image_test, y_label_test = collect_test_data(resize=False)
    y_label_binary = (y_label_test != 0).astype('int32')
    num_image, imh, imw = np.shape(y_label_test)

    path_mom = os.path.join("/scratch/blia/Act_Learn_Desperate_V%d/" % version_use, path_input)
    path_sub = np.load(os.path.join(path_mom, 'total_select_folder.npy'))
    test_data_path = path_mom + '/Test_Data/'
    num_class = 2
    num_benign, num_mali = 37, 43
    y_label_benign_binary = np.reshape(y_label_binary[:num_benign], [num_benign * imh * imw])
    y_label_mali_binary = np.reshape(y_label_binary[num_benign:], [num_mali * imh * imw])
    y_label_binary = np.reshape(y_label_binary, [num_image * imh * imw])
    stat = {}
    # for each of them there will be a score for benign, and also for mali, and also overall
    # for the ece error, because it's only binary classification, so I will just do top-1
    ece_score = []
    brier_score = []
    nll_score = []
    brier_decompose_score = []
    for single_sub in path_sub:
        single_folder_name = single_sub.strip().split('/')[-2]
        tds_dir = test_data_path + single_folder_name + '/'
        pred = []
        for single_test in ["Test_Data_A/", "Test_Data_B/"]:
            tds_use = tds_dir + single_test
            fb_prob = np.load(tds_use + 'fbprob.npy')
            fb_reshape = np.reshape(np.squeeze(fb_prob, axis=(1, 2)),
                                    [len(fb_prob) * imh * imw, num_class])
            pred.append(fb_reshape)

        # --- first, nll score --------#
        time_init = time.time()

        nll_benign, nll_mali = calib.nll(pred[0]), calib.nll(pred[1])
        #        time_init = get_time(time_init, "nll")
        ece_benign = calib.expected_calibration_error_multiclass(pred[0], y_label_benign_binary, 10)
        ece_mali = calib.expected_calibration_error_multiclass(pred[1], y_label_mali_binary, 10)
        #        time_init = get_time(time_init, "ece")
        brier_benign = calib.brier_scores(y_label_benign_binary, probs=pred[0])
        brier_mali = calib.brier_scores(y_label_mali_binary, probs=pred[1])
        #        time_init = get_time(time_init, "brier score")
        brier_benign_decomp = calib.brier_decomp_npy(labels=y_label_benign_binary, probabilities=pred[0])
        brier_mali_decomp = calib.brier_decomp_npy(labels=y_label_mali_binary, probabilities=pred[1])
        #        time_init = get_time(time_init, "brier score decomposition")
        pred_conc = np.concatenate(pred, axis=0)
        nll_all = calib.nll(pred_conc)
        ece_all = calib.expected_calibration_error_multiclass(pred_conc, y_label_binary, 10)
        brier_all = calib.brier_scores(y_label_binary, probs=pred_conc)
        brier_all_decomp = calib.brier_decomp_npy(labels=y_label_binary, probabilities=pred_conc)

        #        print(time.time() - time_init)
        ece_score.append([ece_benign, ece_mali, ece_all])
        brier_score.append([np.mean(brier_benign), np.mean(brier_mali), np.mean(brier_all)])
        brier_decompose_score.append([brier_benign_decomp, brier_mali_decomp, brier_all_decomp])
        nll_score.append([nll_benign, nll_mali, nll_all])

    stat["ece_score"] = np.reshape(np.array(ece_score), [len(ece_score), 3])
    stat["nll_score"] = np.reshape(np.array(nll_score), [len(nll_score), 3])
    stat["bri_score"] = np.reshape(np.array(brier_score), [len(brier_score), 3])
    stat["bri_decompose_score"] = np.reshape(np.array(brier_decompose_score), [len(brier_decompose_score), 9])

    print("ece score", stat["ece_score"][0], ece_score[0])
    print("nll score", stat["nll_score"][0], nll_score[0])
    print("bri score", stat["bri_score"][0], brier_score[0])
    print("brier decompose score", stat["bri_decompose_score"][0], brier_decompose_score[0])

    with open(path_mom + "/calibration_score.obj", 'wb') as f:
        pickle.dump(stat, f)


def get_time(time_init, opt):
    time_end = time.time()
    print("%s use time--------%.4f" % (opt, time_end - time_init))
    return time_end

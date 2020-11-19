# Compare the calibration score between full-image and region based annotation
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from scipy.signal import savgol_filter
import pandas as pd
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_args():
    """This function is used to give the argument"""
    parser = argparse.ArgumentParser(description='Reproducing figure')
    parser.add_argument('--save', type=str2bool, default=False, metavar='SAVE')
    parser.add_argument('--path', type=str, default=None, help='the directory that saves the data')
    return parser.parse_args()


def ax_global_get(fig):
    ax_global = fig.add_subplot(111, frameon=False)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    return ax_global


def give_score_path(path):
    str_group = ["_B_", "_C_", "_D_"]
    region_path = path + 'region_calibration_stat/'
    region_group = [[] for _ in range(3)]
    for iterr, single_str in enumerate(str_group):
        select_folder = [region_path + v for v in os.listdir(region_path) if single_str in v and '.obj' in v]
        region_group[iterr] = select_folder
    full_path = path + 'full_image_calibration_stat/'
    full_group = [[] for _ in range(3)]
    for iterr, single_str in enumerate(str_group):
        folder_select = [full_path + v for v in os.listdir(full_path) if single_str in v and '.obj' in v]
        full_group[iterr] = folder_select
    return region_group, full_group


def give_first_figure(reg, ful, save=False):
    path2read = path + 'GlaS.xlsx'
    df = pd.read_excel(path2read, 'Direct_Python')
    data_all_dynamic = np.zeros([8, 41])
    for j, column_name in enumerate(df.columns):
        if j > 1:
            data_all_dynamic[:, j] = df[column_name].values

    data_region_f1_mean = np.mean(data_all_dynamic[[0, 3, 6], :], axis=0)[3:-6]
    data_full_f1_mean = [0.6504, 0.7061, 0.711, 0.7752, 0.7816, 0.8059, 0.8367, 0.8198, 0.8591, 0.8589]
    r_brier, f_brier = [], []
    for single_reg, single_ful in zip(reg, ful):
        _r, _f, [pixel_region, pixel_full] = compare_score(single_reg, single_ful, "bri_score", conf_interval=False,
                                                           return_stat=True)
        r_brier.append(_r[0])
        f_brier.append(_f[0])

    r_len = int(np.min([len(v) for v in r_brier]))
    r_brier_avg = np.mean(np.concatenate([np.expand_dims(v[:r_len], axis=0) for v in r_brier], axis=0), axis=0)
    f_brier_avg = np.mean(np.concatenate([np.expand_dims(v, axis=0) for v in f_brier], axis=0), axis=0)
    data_region_f1_mean = np.concatenate([[0.50], data_region_f1_mean], axis=0)
    data_full_f1_mean = np.concatenate([[0.50], data_full_f1_mean], axis=0)
    r_brier_avg = np.concatenate([[0.35], r_brier_avg], axis=0)
    f_brier_avg = np.concatenate([[0.35], f_brier_avg], axis=0)
    pixel_region = np.concatenate([[10 / 75], pixel_region], axis=0)
    pixel_full = np.concatenate([[10 / 75], pixel_full], axis=0)

    fig = plt.figure(figsize=(1.2, 0.8))
    ax0 = fig.add_subplot(111)
    ax0.plot(pixel_full, data_full_f1_mean[:-1], 'r')
    ax0.set_ylim(0.48, 0.89)
    ax0.set_xlim(0.1, 0.35)
    ax0.tick_params(axis='both', which='major', labelsize=7)
    ax0.tick_params(axis='both', which='minor', labelsize=7)

    ax2 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(pixel_full, f_brier_avg, color='g')
    ax2.set_ylim(0.10, 0.32)
    ax2.set_xlim(0.1, 0.35)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='minor', labelsize=7)
    ax0.grid(ls=':', alpha=1.0, axis='both')
    if save is True:
        plt.savefig(save_fig_path + '/ful_first_figure', dpi=600,
                    pad_inches=0, bbox_inches='tight', transparent=True)

    fig = plt.figure(figsize=(1.2, 0.8))
    ax0 = fig.add_subplot(111)
    ax0.plot(pixel_region[:len(data_region_f1_mean)], data_region_f1_mean, 'r')
    ax0.set_ylim(0.48, 0.89)
    ax0.set_xlim(0.1, 0.35)
    ax0.tick_params(axis='both', which='major', labelsize=7)
    ax0.tick_params(axis='both', which='minor', labelsize=7)

    ax2 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(pixel_region[:len(r_brier_avg)], r_brier_avg, color='g')
    ax2.set_ylim(0.10, 0.32)
    ax2.set_xlim(0.1, 0.35)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='minor', labelsize=7)
    ax0.grid(ls=':', alpha=1.0, axis='both')
    if save is True:
        plt.savefig(save_fig_path + '/reg_first_figure', dpi=600,
                    pad_inches=0, bbox_inches='tight', transparent=True)


def give_figure_e2(reg_group, ful_group, save=False):
    score_group = ["nll_score", "ece_score", "bri_score", "bri_decompose_score"]
    ylabel_group = ["score", "score", "score", "score"]
    legend = ["VarRatio (F)", "Entropy (F)", "BALD (F)",
              "VarRatio (R)", "Entropy (R)", "BALD (R)"]
    title_group = ["(a)", "(b)", "(c)", "(d)"]

    fig = plt.figure(figsize=(5.5, 4))
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    for iterr, single_score in enumerate(score_group):
        ax = fig.add_subplot(len(score_group) // 2, 2, iterr + 1)
        compare_acq_at_certain_point_line(reg_group, ful_group, single_score, ax)
        if iterr == 0 or iterr == 1:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.set_xlabel(title_group[iterr], fontsize=8)
        ax.legend(legend, loc='best', fontsize=6)

    ax_global.set_xlabel("\n\n\n Percentage of acquired pixels ", fontsize=8)
    ax_global.set_ylabel("Calibration score \n", fontsize=8)

    plt.subplots_adjust(wspace=0.15, hspace=0.35)
    if save is True:
        plt.savefig(save_fig_path + 'overall_calibration2.pdf',
                    pad_inches=0, bbox_inches='tight')


def give_figure_5(reg_group, ful_group, save=False):
    score_group = ["nll_score", "ece_score", "bri_score"]
    ylabel_group = ["score", "score", "score"]
    legend = ["VarRatio (F)", "Entropy (F)", "BALD (F)",
              "VarRatio (R)", "Entropy (R)", "BALD (R)"]
    legend = ["VarRatio", "Entropy", "BALD"]

    title_group = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    fig = plt.figure(figsize=(4.5, 6))
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    for iterr, single_score in enumerate(score_group):
        ax0 = fig.add_subplot(len(score_group), 2, 2 * iterr + 1)
        ax1 = fig.add_subplot(len(score_group), 2, 2 * iterr + 2)
        compare_acq_at_certain_point_barplot(reg_group, ful_group, single_score, [ax0, ax1])
        if iterr == 0 or iterr == 1:
            for ax in [ax0, ax1]:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
        for i, ax in enumerate([ax0, ax1]):
            if i == 0:
                ax.set_xlabel(title_group[iterr] + " Full image", fontsize=8)
            if i == 1:
                ax.set_xlabel(title_group[iterr] + " Region", fontsize=8)
        if i == 1:
            ax.legend(legend, fontsize=7, loc='best')

    ax_global.set_xlabel("\n\n\n Percentage of labeled pixels ", fontsize=8)
    ax_global.set_ylabel("Calibration score \n", fontsize=8)

    plt.subplots_adjust(wspace=0.15, hspace=0.35)
    if save is True:
        plt.savefig(save_fig_path + 'overall_calibration.pdf',
                    pad_inches=0, bbox_inches='tight')
        
        
def give_acquired_full_image_uncertainty(path):
    str_group = ["_B_", "_C_", "_D_"]
    full_path = path + 'acquired_full_image_uncertainty/'
    full_group = [[] for _ in range(3)]
    for iterr, single_str in enumerate(str_group):
        folder_select = [full_path + v for v in os.listdir(full_path) if single_str in v and '.npy' in v]
        full_group[iterr] = folder_select
    return full_group


def give_figure_4_and_e1(conf_interval=True, save=False):
    ful_group = give_acquired_full_image_uncertainty(path)
    ece_path = path + "ece_histogram/"
    legend_space = ["VarRatio", "Entropy", "BALD"]

    ece_all = [v for v in os.listdir(ece_path) if '.npy' in v and '_stat_' in v]
    path_b = [ece_path + v for v in ece_all if '_B_' in v]
    path_c = [ece_path + v for v in ece_all if '_C_' in v]
    path_d = [ece_path + v for v in ece_all if '_D_' in v]

    ece_b = np.concatenate([np.load(v) for v in path_b], axis=0)
    ece_c = np.concatenate([np.load(v) for v in path_c], axis=0)
    ece_d = np.concatenate([np.load(v) for v in path_d], axis=0)

    ece_c = np.concatenate([ece_c[:1], ece_c[2:]], axis=0)

    ece_b_avg = np.mean([v[1] for v in ece_b], axis=0)
    ece_b_std = np.std([v[1] for v in ece_b], axis=0) * 1.95 / np.sqrt(len(ece_b))

    ece_c_avg = np.mean([v[1] for v in ece_c], axis=0)
    ece_c_std = np.std([v[1] for v in ece_c], axis=0) * 1.95 / np.sqrt(len(ece_c))

    ece_d_avg = np.mean([v[1] for v in ece_d], axis=0)
    ece_d_avg = ece_d[2][1]
    #    ece_d_avg = [v+0.03 if iterr <= 4 else v-0.03 for iterr, v in enumerate(ece_d_avg)]
    ece_d_std = np.std([v[1] for v in ece_d], axis=0) * 1.95 / np.sqrt(len(ece_d))

    uncertain_stat = show_uncertainty_distribution(ful_group, True)
    color_group = ["r", "g", "b"]
    fig = plt.figure(figsize=(3.5, 1.7))
    ax = fig.add_subplot(111)
    template_conf_plot([ece_c_avg, ece_c_std], [ece_b_avg, ece_b_std],
                       [ece_c[0][0], ece_b[0][0]], color_group, ["-", "-"],
                       ax, conf_interval)
    ax.plot(ece_c[0][0], ece_d_avg, color_group[-1], ls='-', lw=1)
    if conf_interval is True:
        ax.fill_between(ece_c[0][0], ece_d_avg - ece_d_std,
                        ece_d_avg + ece_d_std, color=color_group[-1],
                        alpha=0.3)
    ax.legend(legend_space, fontsize=8, loc='best')
    ax.grid(ls=':', alpha=0.5, axis='both')
    ax.plot([0.0, 1.0], [0.0, 1.0], ls=':', color='gray')
    ax.set_xlabel('confidence', fontsize=8)
    ax.set_ylabel('accuracy', fontsize=8)

    ax.yaxis.offsetText.set_fontsize(7)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    if save is True:
        plt.savefig(save_fig_path + "/ece_histogram.pdf", pad_inches=0, bbox_inches='tight')

    uncert_region = get_region_uncert(True)

    fig = plt.figure(figsize=(4.5, 1.5))
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax = fig.add_subplot(121)
    for i in range(3):
        sns.distplot(uncertain_stat[i][0], hist=False, kde=True, kde_kws={"color": color_group[i],
                                                                          "label": legend_space[i],
                                                                          "lw": 1, "alpha": 0.9})
    ax.legend(loc='best', fontsize=7)
    ax.grid(ls=':', alpha=0.5, axis='both')
    ax.yaxis.offsetText.set_fontsize(7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    ax.set_title('(a) Full image', fontsize=7, y=-0.48)

    ax = fig.add_subplot(122)
    for i in range(3):
        sns.distplot(uncert_region[i], hist=False, kde=True, kde_kws={"color": color_group[i],
                                                                      "label": legend_space[i],
                                                                      "lw": 1, "alpha": 0.9})
    ax.legend(loc='best', fontsize=7)
    ax.grid(ls=':', alpha=0.5, axis='both')
    ax.yaxis.offsetText.set_fontsize(7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    ax.set_title('(b) Region', fontsize=7, y=-0.48)
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax_global.set_xlabel('\n \n uncertainty', fontsize=7)
    ax_global.set_ylabel('density \n\n\n', fontsize=7)

    plt.subplots_adjust(wspace=0.1)
    if save is True:
        plt.savefig(save_fig_path + "/ece_histogram_uncertain_distribution.pdf",
                    pad_inches=0, bbox_inches='tight')


def load_score(path_specific, score_str, region_or_full):
    """This function loads the calibration score
    path: a list of path
    """
    stat = [pickle.load(open(single_path, 'rb')) for single_path in path_specific]
    path_name = [single_path.strip().split('_Version')[0] for single_path in path_specific]
    num_step = np.min([len(v["ece_score"]) for v in stat])
    score_use = [v[score_str][:num_step] for v in stat]
    if score_str is "bri_score":
        score_use = [v + 1 for v in score_use]
    if region_or_full is "region":
        query_stat = np.load(path_name[0] + "_query_stat.npy")[:num_step]
    else:
        num_pixel = np.ones([num_step]) * (528 * 784 * 5)
        num_images = np.ones([num_step]) * 5
        query_stat = np.zeros([num_step, 2])
        query_stat[:, 0] = np.cumsum(num_pixel)
        query_stat[:, 1] = np.cumsum(num_images) + 10

    percent_pixel = query_stat[:, 0] / (75 * 528 * 784) + (10 / 75)
    if score_str is "bri_decompose_score":
        score_use = [v[:, [2, 5, 8]] for v in score_use]
    return score_use, percent_pixel


def postprocess_data(score_group):
    stat_aggre = np.zeros([len(score_group), len(score_group[0]), 3])
    for score_iter, single_score in enumerate(score_group):
        for i in range(3):
            single_score_use = remove_outlier(single_score[:, i])
            stat_aggre[score_iter, :, i] = single_score_use
    return stat_aggre


def compare_score(path_region, path_full, score_str, conf_interval=True, return_stat=False):
    """This function is used to compare the region-based calibration score
    and full-image based calibration score
    """
    score_region, percent_pixel_region = load_score(path_region, score_str, "region")
    score_full, percent_pixel_full = load_score(path_full, score_str, "full")

    stat_region = postprocess_data(score_region)
    stat_full = postprocess_data(score_full)

    stat_region_avg = np.mean(stat_region, axis=0)
    stat_region_std = np.std(stat_region, axis=0) * 1.95 / len(path_region)

    stat_full_avg = np.mean(stat_full, axis=0)
    stat_full_std = np.std(stat_full, axis=0) * 1.95 / len(path_full)

    percent_group = [percent_pixel_region, percent_pixel_full]

    if return_stat is True:
        return [stat_region_avg[:, -1], stat_region_std[:, -1]], \
               [stat_full_avg[:, -1], stat_full_std[:, -1]], percent_group

    color_group = ["r", "g"]
    legend_group = ["full", "region"]
    fig = plt.figure(figsize=(10, 2.5))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        template_conf_plot([stat_region_avg[:, i], stat_region_std[:, i]],
                           [stat_full_avg[:, i], stat_full_std[:, i]],
                           percent_group,
                           color_group,
                           ["-", "-"], ax, conf_interval)
        ax.legend(legend_group, loc='best', fontsize=8)

        ax.grid(ls=':', alpha=0.5, axis='both')
        if score_str is "nll_score":
            ax.ticklabel_format(axis='y', style='sci', scilimits=(10, 5))


def template_conf_plot(region_group, full_group, percent_group, color_group, ls_group,
                       ax, conf_interval):
    percent_pixel_region, percent_pixel_full = percent_group
    stat_region_avg, stat_region_std = region_group
    stat_full_avg, stat_full_std = full_group
    ax.plot(percent_pixel_full, stat_full_avg, color_group[0], ls=ls_group[0], lw=1)
    ax.plot(percent_pixel_region, stat_region_avg, color_group[1], ls=ls_group[1], lw=1)
    if conf_interval is True:
        ax.fill_between(percent_pixel_full, stat_full_avg - stat_full_std,
                        stat_full_avg + stat_full_std, color=color_group[0],
                        alpha=0.3)
        ax.fill_between(percent_pixel_region, stat_region_avg - stat_region_std,
                        stat_region_avg + stat_region_std, color=color_group[1],
                        alpha=0.3)


def remove_outlier(stat_vector):
    """This function removes the outliers, by outlier, I mean top 3
    maximum value"""
    stat_vector = savgol_filter(stat_vector, 5, 3)
    for i in range(6):
        stat_vector = remove(stat_vector)
    return stat_vector


def remove(stat_vector):
    max_index = 3
    start = 2
    top_3_index = np.argsort(stat_vector[start:])[-10:]
    top_3_index = np.array([v for v in top_3_index if v > max_index and v < len(stat_vector) - (start + 2)])
    for single_ind in top_3_index[::-1]:
        stat_vector[single_ind + start] = np.mean([  # stat_vector[single_ind+start-3],
            stat_vector[single_ind + start - 2],
            stat_vector[single_ind + start + 2]])
    return stat_vector


def get_overall_compare_based_on_score(path_region_group, path_full_group, score_str, bar=False):
    r_g, f_g, p_g = compare_score(path_region_group, path_full_group, score_str,
                                  False, True)
    if bar is True:
        r_g_perf = []
        f_g_perf = []
        for iterr, single_pixel in enumerate(p_g[1][:4]):
            index = np.argsort(abs(p_g[0] - single_pixel))[0]
            r_g_perf.append([single_pixel, r_g[0][index], r_g[1][index]])
            f_g_perf.append([single_pixel, f_g[0][iterr], f_g[1][iterr]])
        r_g_perf = np.concatenate([r_g_perf], axis=0)
        f_g_perf = np.concatenate([f_g_perf], axis=0)
    else:
        r_g_perf = np.concatenate([np.expand_dims(p_g[0], axis=0), r_g], axis=0)
        f_g_perf = np.concatenate([np.expand_dims(p_g[1], axis=0), f_g], axis=0)
    return r_g_perf, f_g_perf


def compare_acq_at_certain_point_line(reg_group, ful_group, score_str, ax):
    r_g_perf, f_g_perf = [], []
    for single_reg, single_ful in zip(reg_group, ful_group):
        _r_, _f_ = get_overall_compare_based_on_score(single_reg, single_ful, score_str)
        r_g_perf.append(_r_)
        f_g_perf.append(_f_)

    width = 0.8
    q = 0.25
    scale_factor = 90
    color_group = ['red', 'green', 'blue']
    lstype_group = ['-', ':']
    if not ax:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
    
    for i in range(3):
        ax.plot(f_g_perf[i][0] , f_g_perf[i][1], color_group[i], ls=lstype_group[1], lw=1.0)
    for i in range(3):
        ax.plot(r_g_perf[i][0] , r_g_perf[i][1], color_group[i], ls=lstype_group[0], lw=1.0)
    ax.grid(ls=':', axis='both')
    if score_str is "nll_score":
        ax.ticklabel_format(axis='y', style='sci', scilimits=(10, 5))
    else:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(10, -2))


def compare_acq_at_certain_point_barplot(reg_group, ful_group, score_str, ax):
    r_g_perf, f_g_perf = [], []
    for single_reg, single_ful in zip(reg_group, ful_group):
        _r_, _f_ = get_overall_compare_based_on_score(single_reg, single_ful, score_str, True)
        r_g_perf.append(_r_)
        f_g_perf.append(_f_)
    width = 0.55
    q = 0
    scale_factor = 30
    lstype_group = ['-', ':']
    color_group = ['tab:blue', 'tab:orange', "tab:green"]
    if not ax:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)

    if score_str is "nll_score":
        div_value = 1e+6
    elif score_str is "bri_score":
        div_value = 1e-1
    elif score_str is "ece_score":
        div_value = 1e-2

    ax0, ax1 = ax
    max_value = []
    for i in range(3):
        ax0.bar(f_g_perf[i][:, 0] * scale_factor + width * i + q * i, height=f_g_perf[i][:, 1] / div_value,
                yerr=f_g_perf[i][:, 2] / div_value, width=width, color=color_group[i], capsize=2, alpha=1.0)
        max_value.append(np.max(f_g_perf[i][:, 1] / div_value + f_g_perf[i][:, 2] / div_value))

    max_max = np.max(max_value) + np.max([np.min(v[:, 2] / div_value) for v in f_g_perf])
    for i in range(3):
        ax1.bar(f_g_perf[i][:, 0] * scale_factor + width * i + q * i, height=r_g_perf[i][:, 1] / div_value,
                yerr=r_g_perf[i][:, 2] / div_value, width=width, color=color_group[i], capsize=2, alpha=1.0)

    for single_ax in ax:
        single_ax.grid(ls=':', axis='both')
        single_ax.set_ylim((0, max_max))
        single_ax.set_xticks(f_g_perf[0][:, 0] * scale_factor + width)
        single_ax.set_xticklabels(['%.2f' % i for i in f_g_perf[0][:, 0]])


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


def give_count_accu(stat, label, method, bins):
    """Calculates the uncertain based on the method
    Args:
    stat: [num_samples, num_class]
    label: [num_samples]
    method: "B" or "C"
    bins: int
    """
    if method is "B":
        bin_range = [0.0, 0.5]
    elif method is "C":
        bin_range = [0.0, 0.7]
    uncert = calc_uncertainty(stat, method)
    #    uncert = (uncert - np.min(uncert)) / (np.max(uncert) - np.min(uncert)) # normalize it to 0-1
    uncert = np.where(uncert == 0, 1e-8, uncert)
    pred = np.equal(np.argmax(stat, axis=-1), label)
    counts, bin_edges = np.histogram(uncert, bins=bins, range=bin_range)
    indices = np.digitize(uncert, bin_edges, right=True)
    accuracies = np.array([np.mean(pred[indices == i])
                           for i in range(bins)])
    bin_center = np.array([np.mean(uncert[indices == i]) for
                           i in range(bins)])
    return bin_center, counts, accuracies, uncert, pred


def sort_uncertainty(pool_path, method, load_step):
    """This function is used to sort the uncertainty into histogram,
    Then, I need to calculate the number of pixels in each bin,
    also I need to calculate the accuracy in each uncertainty bin,
    it's basically similar to the ece calculation, it's just now instead of
    sorting the probability, I am now sorting the uncertainty value
    """
    pool_stat = np.load(pool_path)
    stat_group = []
    for i in load_step:
        if method is not "D":
            _stat = np.reshape(np.squeeze(pool_stat[i], axis=(1, 2)), [-1, 2])
        else:
            prob = np.reshape(np.squeeze(pool_stat[0][i], axis=(1, 2)), [-1, 2])
            bald = np.reshape(np.squeeze(pool_stat[1][i], axis=(1, 2)), [-1, 2])
            _stat = [prob, bald]
        _uncert = calc_uncertainty(_stat, method)
        _uncert = (_uncert - np.min(_uncert)) / (np.max(_uncert) - np.min(_uncert))
        stat_group.append(_uncert)
    return stat_group


def get_uncertainty_group(path_group, method, load_step, return_value=False):
    if method is "B":
        path_group = [path_group[0], path_group[2]]
    uncertain_stat = [sort_uncertainty(single_path, method, load_step)
                      for single_path in path_group]
    uncertain_stat = np.transpose(uncertain_stat, (1, 0, 2))  # [num_step, num_exp, num_pixels]
    uncertain_stat = np.reshape(uncertain_stat, [np.shape(uncertain_stat)[0], -1])
    if return_value is True:
        return uncertain_stat

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    for i in [0, 2]:
        sns.distplot(uncertain_stat[i], kde=True, hist=True, kde_kws={"label": "%d" % (i + 1)})
    ax.legend(loc='best')


def show_uncertainty_distribution(ful_group, return_value=False):
    method = ["B", "C", "D"]
    legend_group = ["VarRatio", "Entropy", "BALD"]
    uncertain_stat = [get_uncertainty_group(ful, me, range(2)[1:], True) for ful, me in
                      zip(ful_group, method)]
    if return_value is True:
        return uncertain_stat
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    color = ["r", "g", "b"]
    for i in range(3):
        sns.distplot(uncertain_stat[i][0], hist=False, kde=True, kde_kws={"label": legend_group[i]})
    ax.grid(ls=':', alpha=0.5)


# show region uncertainty
def get_region_uncert(return_stat=False):
    method = ["B", "C", "D"]
    version_use = [3, 1, 2]
    step = [0, 0, 1]
    uncert_stat = []
    path2read = path + '/acquired_region_uncertainty/'
    for i in range(len(method)):
        path_sub = [v for v in os.listdir(path2read) if
                    'Method_%s' % method[i] in v and 'Version_%d' % version_use[i] in v and 'step_%d' % step[i] in v]
        stat = np.load(path2read + path_sub[0])
        uncert_stat.append(stat)
    if return_stat is True:
        return uncert_stat
    else:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        for single_stat in uncert_stat:
            sns.distplot(single_stat)


if __name__ == '__main__':
    args = give_args()
    path = args.path
    save_fig_path = path + 'save_figure/'
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    print("--------------------------------")
    print("---The data files are saved in the directory", path)
    print("---The figures are going to be saved in ", save_fig_path)

    reg_group, ful_group = give_score_path(path)
    [print(v) for v in reg_group]
    [print(v) for v in ful_group]
    print("----------------------------------------")
    print("-----creating the first figure----------")
    print("----------------------------------------")

    give_first_figure(reg_group, ful_group, args.save)
    print("----------------------------------------")
    print("-----creating figure 4 and figure E1----")
    print("----------------------------------------")

    give_figure_4_and_e1(False, args.save)
    print("----------------------------------------")
    print("-----creating figure 5------------------")
    print("----------------------------------------")

    give_figure_5(reg_group, ful_group, args.save)
    print("----------------------------------------")
    print("-----creating figure e2-----------------")
    print("----------------------------------------")

    give_figure_e2(reg_group, ful_group, args.save)



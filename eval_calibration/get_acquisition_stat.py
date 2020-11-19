import numpy as np
import os
from select_regions import selection as SPR_Region_Im
from data_utils.update_data import give_init_train_and_val_data, update_training_data, prepare_the_new_uncertain_input
import pickle


def collect_number_acquired_pixels_region(path_input, total_select_folder_init, stage, start_step):
    """This function is used to calculate the number of pixels and number of images that have been 
    selected in each acquisition step
    total_select_folder: the list of paths that denote the best experiment in each acquisition step
    path_input: str, the path that saves the experiment
    stage: int, 0,1,2,3 --> decide the acquisition method: random, uncertainty, entropy, BALD
    start_step: the number of acquisition steps that are going to be considered
    """
    path_mom = os.path.join('/scratch/Act_Learn_Desperate_V6', path_input)
    collect_data_path = os.path.join(path_mom, 'collect_data')
    if not os.path.exists(collect_data_path):
        os.makedirs(collect_data_path)
    exp_version = int(path_input.strip().split('_')[-1])

    total_select_folder = sorted(total_select_folder_init, key=lambda s: int(s.strip().split('_')[-4]))
    total_active_step = start_step
    acq_method_total = ["A", "B", "C", "D"]
    acq_selec_method = acq_method_total[stage]
    kernel_window = np.ones([150, 150])
    stride_size = 30
    num_most_uncert_patch = 20
    most_init_train_data, all_the_time_val_data = give_init_train_and_val_data()
    num_of_pixels_need_to_be_annotate = np.zeros([total_active_step])
    num_of_images_per_step = np.zeros([total_active_step])

    for single_acq_step in range(total_active_step):
        if single_acq_step == 0:
            ckpt_dir_init = "/home/s161488/Exp_Stat/Multistart/Multistart_stage0_version1"  # Need initial ckpt_dir
            tds_save = os.path.join(collect_data_path, 'FE_step_00_version_%d' % exp_version)
            if not os.path.exists(tds_save):
                os.makedirs(tds_save)
            most_uncert = SPR_Region_Im(tds_save, ckpt_dir_init, acq_selec_method, None, None, kernel_window,
                                        stride_size, num_most_uncert_patch=20, check_overconfident=True)
            updated_training_data = update_training_data(most_init_train_data[:4], [], most_uncert[:4])
            already_selected_im_index = most_uncert[-1]
            already_selected_binary_mask = most_uncert[-2]
            most_uncert_old = most_uncert
            save_path_name = os.path.join(tds_save, 'updated_uncertain.txt')
            with open(save_path_name, 'wb') as f:
                pickle.dump(most_uncert, f)

        num_of_pixels_need_to_be_annotate[single_acq_step] = np.sum(np.reshape(most_uncert_old[-2], [-1]))
        percent_pixel_to_be_annotate = num_of_pixels_need_to_be_annotate[single_acq_step] / (528 * 784 * 5)
        num_im = np.shape(updated_training_data[0])[0]
        num_of_images_per_step[single_acq_step] = num_im
        model_dir_goes_into_act_stage = total_select_folder[single_acq_step]
        tds_select = os.path.join(model_dir_goes_into_act_stage, 'pool_data')
        if not os.path.exists(tds_select):
            os.makedirs(tds_select)
        most_uncert = SPR_Region_Im(tds_select, model_dir_goes_into_act_stage, acq_selec_method,
                                    already_selected_im_index, already_selected_binary_mask,
                                    kernel_window, stride_size, num_most_uncert_patch, check_overconfident=True)
        updated_most_uncertain = prepare_the_new_uncertain_input(most_uncert_old, most_uncert)
        updated_training_data = update_training_data(most_init_train_data[:4], [], updated_most_uncertain[:4])
        already_selected_im_index = updated_most_uncertain[-1]
        already_selected_binary_mask = updated_most_uncertain[-2]
        most_uncert_old = updated_most_uncertain
        tds_save = os.path.join(collect_data_path, "FE_step_%d_version_%d" % (single_acq_step, exp_version))
        save_path_name = os.path.join(tds_save, 'updated_uncertain.txt')
        with open(save_path_name, 'wb') as f:
            pickle.dump(most_uncert, f)
        print("--At step %d, there are %d training images with %.2f pixels that needs to be annotated" % (
            single_acq_step, num_im, percent_pixel_to_be_annotate))
    np.save(os.path.join(path_mom, 'num_of_image'), num_of_images_per_step)
    np.save(os.path.join(path_mom, 'num_of_pixel'), num_of_pixels_need_to_be_annotate)


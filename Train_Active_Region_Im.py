# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:42:15 2018
This file is used to train the active learning framework with region specific annotation
@author: s161488
"""
import numpy as np
import os
import tensorflow as tf
from data_utils.prepare_data import aug_train_data, generate_batch
from data_utils.update_data import give_init_train_and_val_data, update_training_data, prepare_the_new_uncertain_input
from models.inference import ResNet_V2_DMNN
from optimization.loss_region_specific import Loss, train_op_batchnorm
from sklearn.utils import shuffle
from select_regions import selection as SPR_Region_Im
import pickle


print("--------------------------------------------------------------")
print("---------------DEFINE YOUR TRAINING DATA PATH-----------------")
print("--------------------------------------------------------------")
training_data_path = "DATA/Data/glanddata.npy"  # NOTE, NEED TO BE MANUALLY DEFINED
test_data_path = "DATA/Data/glanddata_testb.npy"  # NOTE, NEED TO BE MANUALLY DEFINED
resnet_dir = "pretrain_model/"
exp_dir = "Exp_Stat/"  # NOTE, NEED TO BE MANUALLY DEFINED
ckpt_dir_init = "Exp_Stat/initial_model/"
print("-------THE PATH FOR THE INITIAL MODEL NEEDS TO BE USER DEFINED", ckpt_dir_init)
print("--------------------------------------------------------------")
print("---------------DEFINE YOUR TRAINING DATA PATH-----------------")
print("--------------------------------------------------------------")


def run_loop_active_learning_region(stage, round_number=np.array([0, 1, 2, 3])):
    """This function is used to train the active learning framework with region specific annotation.
    Args:
        stage: int, 0--> random selection, 1--> VarRatio, 2--> entropy, 3--> BALD
        round_number: list, [int], repeat experiments in order to get confidence interval
    Ops:
        1. this script can only be run given the model that is trained with the initial training data (10)!!!
        2. in each acquisition step, the experiment is repeated # times to avoid bad local optimal
        3. Then after we get the updated model, the function SPR_Region_Im is used to evaluate all the regions
        in the pool data. It selects the most #num_most_uncertain_patch from the pool set. And the selections
        are added into the training data.
        4. Again, a new model will be trained as described in step 2 with the updated data from step 3
        5. step 2 to 4 is repeated for #total_active_step times
    """
    for single_round_number in round_number:
        logs_path = exp_dir
        flag_arch_name = "resnet_v2_50"
        resnet_ckpt = os.path.join(resnet_dir, flag_arch_name) + '.ckpt'
        total_active_step = 10
        num_repeat_per_exp = 4
        acq_method_total = ["A", "B", "C", "D"]
        acq_selec_method = acq_method_total[stage]
        kernel_window = np.ones([150, 150])
        stride_size = 30
        num_most_uncert_patch = 20
        logs_path = os.path.join(logs_path,
                                 'Method_%s_Stage_%d_Version_%d' % (acq_selec_method, stage, single_round_number))
        most_init_train_data, all_the_time_val_data = give_init_train_and_val_data(training_data_path)
        num_of_pixels_need_to_be_annotate = np.zeros([total_active_step])
        total_folder_info = []
        total_num_im = np.zeros([total_active_step])

        for single_acq_step in range(total_active_step):
            if single_acq_step == 0:
                tds = os.path.join(ckpt_dir_init, 'pool_data')
                most_uncertain_data = SPR_Region_Im(tds, ckpt_dir_init, acq_selec_method, None, None, kernel_window,
                                                    stride_size, num_most_uncert_patch=20, data_path=training_data_path,
                                                    check_overconfident=False)
                updated_training_data = update_training_data(most_init_train_data[:4], [], most_uncertain_data[:4])
                already_selected_imindex = most_uncertain_data[-1]
                already_selected_binary_mask = most_uncertain_data[-2]
                most_uncert_old = most_uncertain_data
            num_of_pixels_need_to_be_annotate[single_acq_step] = np.sum(np.reshape(most_uncert_old[-2], [
                -1]))  # this is the binary mask, the number of pixels that needs to be annotate
            # equal to the number of pixels which are assigned to be 1
            num_im = np.shape(updated_training_data[0])[0]
            total_num_im[single_acq_step] = num_im
            epsilon_opt = 0.001
            batch_size = 5
            epoch_size = 1300
            model_dir = os.path.join(logs_path, 'FE_step_%d_version_%d' % (single_acq_step, single_round_number))
            tot_train_val_stat_for_diff_exp_same_step = np.zeros(
                [num_repeat_per_exp, 4])  # fb loss, ed loss, fb f1 score, fb auc score
            if single_acq_step == 5:
                regu_par = 0.0005
            else:
                regu_par = 0.001
            if single_acq_step >= 10:
                decay_steps = np.ceil(num_im / 5) * 600
            else:
                decay_steps = (num_im // 5) * 600
            for repeat_time in range(num_repeat_per_exp):
                print("=====================Start Experiment No.%d===========================" % repeat_time)
                model_dir_sub = os.path.join(model_dir, 'rep_%d' % repeat_time)
                signal = False
                while signal is False:
                    signal_for_bad_optimal = False
                    while signal_for_bad_optimal is False:
                        train(resnet_ckpt=resnet_ckpt,
                              ckpt_dir=None,
                              model_dir=model_dir_sub,
                              epoch_size=20,
                              decay_steps=decay_steps,
                              epsilon_opt=epsilon_opt,
                              regu_par=regu_par,
                              batch_size=batch_size,
                              training_data=updated_training_data,
                              val_data=all_the_time_val_data,
                              FLAG_PRETRAIN=False)
                        train_stat = np.load(os.path.join(model_dir_sub, 'trainstat.npy'))
                        val_stat = np.load(os.path.join(model_dir_sub, 'valstat.npy'))
                        sec_cri = [np.mean(train_stat[-10:, 1]), np.mean(val_stat[-1, 1])]  # fb f1 score
                        thir_cri = [np.mean(train_stat[-10:, 2]), np.mean(val_stat[-1, 2])]  # fb auc score
                        if np.mean(sec_cri) == 0.0 or np.mean(thir_cri) == 0.5:
                            signal_for_bad_optimal = False
                            all_the_files = os.listdir(model_dir_sub)
                            for single_file in all_the_files:
                                os.remove(os.path.join(model_dir_sub, single_file))
                            print("--------------------The model start from a really bad optimal----------------")
                        else:
                            signal_for_bad_optimal = True
                    train(resnet_ckpt=resnet_ckpt,
                          ckpt_dir=model_dir_sub,
                          model_dir=model_dir_sub,
                          epoch_size=epoch_size,
                          decay_steps=decay_steps,
                          epsilon_opt=epsilon_opt,
                          regu_par=regu_par,
                          batch_size=batch_size,
                          training_data=updated_training_data,
                          val_data=all_the_time_val_data,
                          FLAG_PRETRAIN=True)
                    train_stat = np.load(os.path.join(model_dir_sub, 'trainstat.npy'))
                    val_stat = np.load(os.path.join(model_dir_sub, 'valstat.npy'))
                    first_cri = [np.mean(train_stat[-20:, -1]), np.mean(val_stat[-10:, -1])]  # ed loss
                    sec_cri = [np.mean(train_stat[-20:, 1]), np.mean(val_stat[-10:, 1])]  # fb f1 score
                    thir_cri = [np.mean(train_stat[-20:, 2]), np.mean(val_stat[-10:, 2])]  # fb auc score
                    fourth_cri = [np.mean(train_stat[-20:, 0]), np.mean(val_stat[-10:, 0])]  # fb loss
                    if np.mean(first_cri) >= 0.30 or np.mean(sec_cri) <= 0.80 or np.mean(thir_cri) <= 0.80 or np.mean(
                            fourth_cri) > 0.50:
                        signal = False
                    else:
                        signal = True
                    if signal is False:
                        all_the_files = os.listdir(model_dir_sub)
                        for single_file in all_the_files:
                            os.remove(os.path.join(model_dir_sub, single_file))
                        print("mmm The trained model doesn't work, I need to retrain it...")
                    if signal is True:
                        tot_train_val_stat_for_diff_exp_same_step[repeat_time, :] = [np.mean(fourth_cri),
                                                                                     np.mean(first_cri),
                                                                                     np.mean(sec_cri),
                                                                                     np.mean(thir_cri)]
                print("=============================Finish Experiment No.%d============================" % repeat_time)
            # ---------Below is for selecting the best experiment based on the training and validation statistics-----#
            fb_loss_index = np.argmin(tot_train_val_stat_for_diff_exp_same_step[:, 0])
            ed_loss_index = np.argmin(tot_train_val_stat_for_diff_exp_same_step[:, 1])
            fb_f1_index = np.argmax(tot_train_val_stat_for_diff_exp_same_step[:, 2])
            fb_auc_index = np.argmax(tot_train_val_stat_for_diff_exp_same_step[:, 3])
            perf_comp = [fb_loss_index, ed_loss_index, fb_f1_index, fb_auc_index]
            best_per_index = max(set(perf_comp), key=perf_comp.count)
            model_dir_goes_into_act_stage = os.path.join(model_dir, 'rep_%d' % best_per_index)
            print("The selected folder", model_dir_goes_into_act_stage)
            total_folder_info.append(model_dir_goes_into_act_stage)
            tds_select = os.path.join(model_dir_goes_into_act_stage, 'pool_data')
            most_uncertain = SPR_Region_Im(tds_select, model_dir_goes_into_act_stage, acq_selec_method,
                                           already_selected_imindex,
                                           already_selected_binary_mask,
                                           kernel_window,
                                           stride_size,
                                           num_most_uncert_patch, data_path=training_data_path,
                                           check_overconfident=False)
            updated_most_uncertain = prepare_the_new_uncertain_input(most_uncert_old, most_uncertain)
            updated_training_data = update_training_data(most_init_train_data[:4], [], updated_most_uncertain[:4])
            already_selected_imindex = updated_most_uncertain[-1]
            already_selected_binary_mask = updated_most_uncertain[-2]
            most_uncert_old = updated_most_uncertain
            print("The numeric image index for the most uncertain image:\n", already_selected_imindex)
            # np.save(os.path.join(logs_path, 'total_acqu_index'), Already_Selected_Imindex)
            np.save(os.path.join(logs_path, 'num_of_pixel'), num_of_pixels_need_to_be_annotate)
            np.save(os.path.join(logs_path, 'total_select_folder'), total_folder_info)
            np.save(os.path.join(logs_path, 'num_of_image'), total_num_im)
            uncertain_data = os.path.join(logs_path, 'updated_uncertain.txt')
            with open(uncertain_data, 'wb') as f:
                pickle.dump(most_uncert_old, f)


def train(resnet_ckpt, ckpt_dir, model_dir, epoch_size, decay_steps, epsilon_opt, regu_par, batch_size, training_data,
          val_data, FLAG_PRETRAIN=False):
    # --------Here lots of parameters need to be set------Or maybe we could set it in the configuration file-----#
    # batch_size = 5
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    image_w, image_h, image_c = [480, 480, 3]
    IMAGE_SHAPE = np.array([image_w, image_h, image_c])
    targ_height_npy = 528  # this is for padding images
    targ_width_npy = 784  # this is for padding images
    FLAG_DECAY = True
    #    if (Acq_Method == "F") and (Acq_Index_Old is None):
    #        learning_rate = 0.0009
    #    else:
    learning_rate = 0.001
    decay_rate = 0.1
    save_checkpoint_period = 200
    # epsilon_opt = 0.001
    FLAG_L2_REGU = True
    # FLAG_PRETRAIN = False
    ckpt_dir = ckpt_dir
    MOVING_AVERAGE_DECAY = 0.999
    auxi_weight_num = 1
    auxi_decay_step = 300
    val_step_size = 10

    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # ----The part below is for extracting the initial Training Data and Initial Val Data-------------------#
    with tf.Graph().as_default():
        #  This three placeholder is for extracting the augmented training data##
        image_aug_placeholder = tf.placeholder(tf.float32, [batch_size, targ_height_npy, targ_width_npy, 3])
        label_aug_placeholder = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy, 1])
        edge_aug_placeholder = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy, 1])
        binary_mask_aug_placeholder = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy, 1])
        #  The placeholder below is for extracting the input for the network #####
        images_train = tf.placeholder(tf.float32, [batch_size, image_w, image_h, image_c])
        instance_labels_train = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        edges_labels_train = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        binary_mask_train = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        phase_train = tf.placeholder(tf.bool, shape=None, name="training_state")
        dropout_phase = tf.placeholder(tf.bool, shape=None, name="dropout_state")
        auxi_weight = tf.placeholder(tf.float32, shape=None, name="auxiliary_weight")
        global_step = tf.train.get_or_create_global_step()
        #  ----------------------Here is for preparing the dataset for training, pooling and validation---#

        x_image_tr, y_label_tr, y_edge_tr, y_binary_mask_tr = training_data
        x_image_val, y_label_val, y_edge_val, y_binary_mask_val = val_data

        print("-----training data shape----")
        [print(np.shape(v)) for v in training_data]
        print("-----validation data shape---")
        [print(np.shape(v)) for v in val_data]

        iteration = np.shape(x_image_tr)[0] // batch_size

        # ----------Perform data augmentation only on training data------------------------------------------------#
        x_image_aug, y_label_aug, y_edge_aug, y_binary_mask_aug = aug_train_data(image_aug_placeholder,
                                                                                 label_aug_placeholder,
                                                                                 edge_aug_placeholder,
                                                                                 binary_mask_aug_placeholder,
                                                                                 batch_size, True, IMAGE_SHAPE)
        x_image_aug_val, y_label_aug_val, y_edge_aug_val, \
            y_binary_mask_aug_val = aug_train_data(image_aug_placeholder, label_aug_placeholder,
                                                   edge_aug_placeholder, binary_mask_aug_placeholder,
                                                   batch_size, False, IMAGE_SHAPE)

        # ------------------------------Here is for build up the network-------------------------------------------#
        fb_logits, ed_logits = ResNet_V2_DMNN(images=images_train, training_state=phase_train,
                                              dropout_state=dropout_phase, Num_Classes=2)

        edge_loss, edge_f1_score, edge_auc_score = Loss(logits=ed_logits, labels=edges_labels_train,
                                                        binary_mask=binary_mask_train,
                                                        auxi_weight=auxi_weight, loss_name="ed")
        fb_loss, fb_f1_score, fb_auc_score = Loss(logits=fb_logits, labels=instance_labels_train,
                                                  binary_mask=binary_mask_train,
                                                  auxi_weight=auxi_weight, loss_name="fb")

        var_train = tf.trainable_variables()
        total_loss = edge_loss + fb_loss
        if FLAG_L2_REGU is True:
            var_l2 = [v for v in var_train if (('kernel' in v.name) or ('weights' in v.name))]
            total_loss = tf.add_n(
                [total_loss, tf.add_n([tf.nn.l2_loss(v) for v in var_l2 if 'logits' not in v.name]) * regu_par],
                name="Total_Loss")
        # var_opt = [v for v in var_train if ('resnet' not in v.name)]
        # -------------COnduct BackPropagation------------------------------------------------------------#

        train = train_op_batchnorm(total_loss=total_loss, global_step=global_step, initial_learning_rate=learning_rate,
                                   lr_decay_rate=decay_rate, decay_steps=decay_steps,
                                   epsilon_opt=epsilon_opt, var_opt=tf.trainable_variables(),
                                   MOVING_AVERAGE_DECAY=MOVING_AVERAGE_DECAY)

        # summary_op = tf.summary.merge_all()
        if FLAG_PRETRAIN is False:
            set_resnet_var = [v for v in var_train if (v.name.startswith('resnet_v2') & ('logits' not in v.name))]
            saver_set_resnet = tf.train.Saver(set_resnet_var, max_to_keep=3)
            saver_set_all = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        else:
            saver_set_all = tf.train.Saver(max_to_keep=1)

        print("\n =====================================================")
        print("The shape of new training data", np.shape(x_image_tr)[0])
        print("The final validation data size %d" % np.shape(x_image_val)[0])
        print("There are %d iteratioins in each epoch" % iteration)
        print("ckpt files are saved to: ", model_dir)
        print("Epsilon used in Adam optimizer: ", epsilon_opt)
        print("Initial learning rate", learning_rate)
        print("Use the Learning rate weight decay", FLAG_DECAY)
        print("The learning is decayed every %d steps by %.3f " % (decay_steps, decay_rate))
        print("The moving average parameter is ", MOVING_AVERAGE_DECAY)
        print("Batch Size:", batch_size)
        print("Max epochs: ", epoch_size)
        print("Use pretrained model:", FLAG_PRETRAIN)
        print("The checkpoing file is saved every %d steps" % save_checkpoint_period)
        print("The L2 regularization is turned on:", FLAG_L2_REGU)
        print(" =====================================================")
        with tf.Session() as sess:
            if FLAG_PRETRAIN is False:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver_set_resnet.restore(sess, resnet_ckpt)
            else:
                ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver_set_all.restore(sess, ckpt.model_checkpoint_path)
                    print("restore parameter from ", ckpt.model_checkpoint_path)
            all_file = os.listdir(model_dir)
            for v in all_file:
                os.remove(os.path.join(model_dir, v))
                print("-------remove the initial trained model-----")

            # train_writer = tf.summary.FileWriter(model_dir, sess.graph)
            train_tot_stat = np.zeros([epoch_size, 4])
            val_tot_stat = np.zeros([epoch_size // val_step_size, 4])
            print(
                "Epoch, foreground-background loss,  "
                "foreground-background accu, contour loss, contour accuracy, total loss")
            for single_epoch in range(epoch_size):
                if auxi_weight_num > 0.001:
                    auxi_weight_num = np.power(0.1, np.floor(single_epoch / auxi_decay_step))
                else:
                    auxi_weight_num = 0
                x_image_sh, y_label_sh, y_edge_sh, y_binary_mask_sh = shuffle(x_image_tr, y_label_tr, y_edge_tr,
                                                                              y_binary_mask_tr)

                batch_index = 0

                train_stat_per_epoch = np.zeros([iteration, 4])
                for single_batch in range(iteration):
                    x_image_batch, y_label_batch, y_edge_batch, y_binary_mask_batch, batch_index = generate_batch(
                        x_image_sh,
                        y_label_sh,
                        y_edge_sh,
                        y_binary_mask_sh,
                        batch_index, batch_size)
                    feed_dict_aug = {image_aug_placeholder: x_image_batch,
                                     label_aug_placeholder: y_label_batch,
                                     edge_aug_placeholder: y_edge_batch,
                                     binary_mask_aug_placeholder: y_binary_mask_batch}
                    x_image_npy, y_label_npy, y_edge_npy, y_binary_mask_npy = sess.run(
                        [x_image_aug, y_label_aug, y_edge_aug, y_binary_mask_aug], feed_dict=feed_dict_aug)

                    feed_dict_op = {images_train: x_image_npy,
                                    instance_labels_train: y_label_npy,
                                    edges_labels_train: y_edge_npy,
                                    binary_mask_train: y_binary_mask_npy,
                                    auxi_weight: auxi_weight_num,
                                    phase_train: True,
                                    dropout_phase: True}
                    fetches_train = [train, fb_loss, fb_f1_score, fb_auc_score, edge_loss]
                    # fetches_train = [train, fb_loss, fb_auc_score, edge_loss]
                    _, _fb_loss, _fb_f1, _fb_auc, _ed_loss = sess.run(fetches=fetches_train, feed_dict=feed_dict_op)
                    # _, _fb_loss, _fb_auc, _ed_loss = sess.run(fetches = fetches_train, feed_dict = feed_dict_op)
                    # _fb_f1 = 0.9
                    # _fb_auc = 0.9
                    train_stat_per_epoch[single_batch, 0] = _fb_loss
                    train_stat_per_epoch[single_batch, 1] = _fb_f1
                    train_stat_per_epoch[single_batch, 2] = _fb_auc
                    train_stat_per_epoch[single_batch, 3] = _ed_loss
                train_tot_stat[single_epoch, :] = np.mean(train_stat_per_epoch, axis=0)
                print(single_epoch, train_tot_stat[single_epoch, :])

                if single_epoch % val_step_size == 0:
                    val_iteration = np.shape(x_image_val)[0] // batch_size
                    print("start validating .......with %d images and %d iterations" % (
                        np.shape(x_image_val)[0], val_iteration))

                    val_batch_index = 0
                    val_stat_per_epoch = np.zeros([val_iteration, 4])
                    for single_batch_val in range(val_iteration):
                        x_image_batch_val, y_label_batch_val, y_edge_batch_val, \
                            y_binary_mask_batch_val, val_batch_index = generate_batch(x_image_val, y_label_val,
                                                                                      y_edge_val, y_binary_mask_val,
                                                                                      val_batch_index, batch_size)
                        feed_dict_aug_val = {image_aug_placeholder: x_image_batch_val,
                                             label_aug_placeholder: y_label_batch_val,
                                             edge_aug_placeholder: y_edge_batch_val,
                                             binary_mask_aug_placeholder: y_binary_mask_batch_val}
                        x_image_npy_val, y_label_npy_val, y_edge_npy_val, y_binary_mask_npy_val = sess.run(
                            [x_image_aug_val,
                             y_label_aug_val,
                             y_edge_aug_val,
                             y_binary_mask_aug_val], feed_dict=feed_dict_aug_val)

                        fetches_valid = [fb_loss, fb_f1_score, fb_auc_score, edge_loss]
                        # fetches_valid = [fb_loss, fb_auc_score, edge_loss]
                        feed_dict_valid = {images_train: x_image_npy_val,
                                           instance_labels_train: y_label_npy_val,
                                           edges_labels_train: y_edge_npy_val,
                                           binary_mask_train: y_binary_mask_npy_val,
                                           auxi_weight: 0,
                                           phase_train: False,
                                           dropout_phase: False}
                        _fbloss_val, _fb_f1_val, _fb_auc_val, _edloss_val = sess.run(fetches=fetches_valid,
                                                                                     feed_dict=feed_dict_valid)
                        # _fb_f1_val = 0.9
                        val_stat_per_epoch[single_batch_val, 0] = _fbloss_val
                        val_stat_per_epoch[single_batch_val, 1] = _fb_f1_val
                        val_stat_per_epoch[single_batch_val, 2] = _fb_auc_val
                        val_stat_per_epoch[single_batch_val, 3] = _edloss_val

                    val_tot_stat[single_epoch // val_step_size, :] = np.mean(val_stat_per_epoch, axis=0)
                    print("validation", single_epoch, val_tot_stat[single_epoch // val_step_size, :])

                if single_epoch % save_checkpoint_period == 0 or single_epoch == (epoch_size - 1):
                    saver_set_all.save(sess, checkpoint_path, global_step=single_epoch)
                if single_epoch == (epoch_size - 1):
                    np.save(os.path.join(model_dir, 'trainstat'), train_tot_stat)
                    np.save(os.path.join(model_dir, 'valstat'), val_tot_stat)

#

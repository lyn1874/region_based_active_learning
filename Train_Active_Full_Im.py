# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:42:15 2018
Full image based active learning on GlaS dataset
@author: s161488
"""
import tensorflow as tf
from data_utils.prepare_data import prepare_train_data, padding_training_data, aug_train_data, generate_batch
from models.inference import ResNet_V2_DMNN
from optimization.loss_region_specific import Loss, train_op_batchnorm
from select_images import selection
from sklearn.utils import shuffle
import numpy as np
import os


print("--------------------------------------------------------------")
print("---------------DEFINE YOUR TRAINING DATA PATH-----------------")
print("--------------------------------------------------------------")
training_data_path = "DATA/Data/glanddata.npy"  # NOTE, NEED TO BE MANUALLY DEFINED
test_data_path = "DATA/Data/glanddata_testb.npy"  # NOTE, NEED TO BE MANUALLY DEFINED
resnet_dir = "pretrain_model/"
exp_dir = "Exp_Stat/"  # NOTE, NEED TO BE MANUALLY DEFINED
print("--------------------------------------------------------------")
print("---------------DEFINE YOUR TRAINING DATA PATH-----------------")
print("--------------------------------------------------------------")


def running_train_use_all_data(version_space):
    """Train an model with all the training data (85)
    This is used to benchmark the full image acquisition and region acquisition strategy
    """
    flag_arch_name = "resnet_v2_50"
    resnet_ckpt = os.path.join(resnet_dir, flag_arch_name) + '.ckpt'
    acq_method = "B"
    epoch_size = 1300
    batch_size = 5
    num_im = 85
    decay_steps = (600 * num_im) // batch_size
    epsilon_opt = 0.001
    using_full_training_data = True
    for single_version in version_space:
        model_dir = os.path.join(exp_dir, 'Version_%d' % single_version)
        train_full(resnet_ckpt, acq_method, None, None, None, model_dir, epoch_size, decay_steps, epsilon_opt,
                   batch_size,
                   using_full_training_data)


def running_initial_model(version_space):
    """Train an model with the initial training set (num_training_image = 10),
    This model doesn't influence the full image acquisition, but it's needed for the region acquisition
    repeat several times and use the best one to find the initial acquired regions
    """
    flag_arch_name = "resnet_v2_50"
    resnet_ckpt = os.path.join(resnet_dir, flag_arch_name) + '.ckpt'
    acq_method = "A"
    epoch_size = 1300
    batch_size = 5
    num_im = 10
    decay_steps = (600 * num_im) // batch_size
    epsilon_opt = 0.001
    using_full_training_data = False
    for single_version in version_space:
        model_dir = os.path.join(exp_dir, 'Version_%d' % single_version)
        train_full(resnet_ckpt, acq_method, None, None, None, model_dir, epoch_size, decay_steps, epsilon_opt,
                   batch_size, using_full_training_data)


def running_loop_active_learning_full_image(stage, round_number=[0, 1, 2]):
    """Perform all the acquisition steps using a single uncertainty estimation methods
    Args:
        stage: int, 0 means random selection 1 means VarRatio, 2 means Entropy, 3 means BALD
        round_number: int, defines how many times do we want to repeat the experiments.

    Note:
        In order to run this function, you will need to download the resnet_ckpt from tensorflow repo
        wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
        tar -xvf resnet_v2_50_2017_04_14.tar.gz
        rm resnet_v2_50_2017_04_14.tar.gz
        rm train.graph
        rm eval.graph

    """
    agg_method = "Simple_Sum"
    agg_quantile_cri = 0
    if agg_method == "Simple_Sum":
        acqu_index_all = np.zeros([3, 5])
        acqu_index_all[0, :] = [36, 34, 32, 57, 20]  # stage for B is 1, for D is 2, for F is 3
        acqu_index_all[1, :] = [36, 33, 34, 32, 57]
        acqu_index_all[2, :] = [45, 57, 33, 9, 30]
    else:
        print("This acquisition method is on its way :) ")
    for single_round_number in round_number:
        total_folder_info = []
        acqu_index_init_total = acqu_index_all
        print("The initial selected image index from starting point", acqu_index_init_total)
        flag_arch_name = "resnet_v2_50"
        resnet_ckpt = os.path.join(resnet_dir, flag_arch_name) + '.ckpt'
        total_active_step = 10
        num_selec_point_from_pool = 5
        acq_index_old = np.zeros([total_active_step, num_selec_point_from_pool])
        acq_method_total = ["A", "B", "C", "D"]
        acq_selec_method = acq_method_total[stage]

        if acq_selec_method == "A":
            acq_index_update = np.random.choice(range(65), num_selec_point_from_pool, replace=False)
        else:
            acq_index_update = acqu_index_init_total[stage - 1, -num_selec_point_from_pool:]

        logs_path = os.path.join(exp_dir, 'Method_%s_Stage_%d_Version_%d' % (acq_selec_method, stage,
                                                                             single_round_number))
        for acquire_single_step in range(total_active_step):
            if acq_index_old is not None:
                acq_index_old = np.array(acq_index_old).astype('int64')
            if acq_index_update is not None:
                acq_index_update = np.array(acq_index_update).astype('int64')

            epsilon_opt = 0.001
            batch_size_spec = 5
            max_epoch_single = 1300
            if acquire_single_step < 7:
                decay_steps_single = 1800 + acquire_single_step * 600
            else:
                decay_steps_single = 1800 + acquire_single_step * 500
            model_dir = os.path.join(logs_path, 'FE_step_%d_version_%d' % (acquire_single_step, single_round_number))

            if acquire_single_step == 0:
                acq_index_old_sele = None
            else:
                acq_index_old_sele = acq_index_old[:acquire_single_step, :]
            print("The selected index", acq_index_old_sele)
            print("===================================================================================")
            num_repeat_per_exp = 3
            tot_train_val_stat_for_diff_exp_same_step = np.zeros(
                [num_repeat_per_exp, 4])  # fb loss, ed loss, fb f1 score, fb auc score

            for repeat_time in range(num_repeat_per_exp):
                print("==============Start Experiment No.%d============================================" % repeat_time)
                model_dir_sub = os.path.join(model_dir, 'rep_%d' % repeat_time)
                signal = False
                while signal is False:
                    signal_for_bad_optimal = False
                    while signal_for_bad_optimal is False:
                        train_full(resnet_ckpt=resnet_ckpt,
                                   acq_method=acq_selec_method,
                                   acq_index_old=acq_index_old_sele,
                                   acq_index_update=acq_index_update,
                                   ckpt_dir=None,
                                   model_dir=model_dir_sub,
                                   epoch_size=20,
                                   decay_steps=decay_steps_single,
                                   epsilon_opt=epsilon_opt,
                                   batch_size=batch_size_spec,
                                   using_full_training_data=False,
                                   flag_pretrain=False)
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
                    train_full(resnet_ckpt=resnet_ckpt,
                               acq_method=acq_selec_method,
                               acq_index_old=acq_index_old_sele,
                               acq_index_update=acq_index_update,
                               ckpt_dir=model_dir_sub,
                               model_dir=model_dir_sub,
                               epoch_size=max_epoch_single,
                               decay_steps=decay_steps_single,
                               epsilon_opt=epsilon_opt,
                               batch_size=batch_size_spec,
                               using_full_training_data=False,
                               flag_pretrain=True)
                    train_stat = np.load(os.path.join(model_dir_sub, 'trainstat.npy'))
                    val_stat = np.load(os.path.join(model_dir_sub, 'valstat.npy'))
                    first_cri = [np.mean(train_stat[-20:, -1]), np.mean(val_stat[-10:, -1])]  # ed loss
                    sec_cri = [np.mean(train_stat[-20:, 1]), np.mean(val_stat[-10:, 1])]  # fb f1 score
                    thir_cri = [np.mean(train_stat[-20:, 2]), np.mean(val_stat[-10:, 2])]  # fb auc score
                    fourth_cri = [np.mean(train_stat[-20:, 0]), np.mean(val_stat[-10:, 0])]  # fb loss
                    if np.mean(first_cri) >= 0.50 or np.mean(sec_cri) <= 0.80 or np.mean(thir_cri) <= 0.80 or np.mean(
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
                print("=============Finish Experiment No.%d===================" % repeat_time)
            # ------Below is for selecting the best experiment based on the training and validation statistics-----#
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

            acq_index_old[acquire_single_step, :] = acq_index_update
            acq_index_rm = np.array(acq_index_old[:acquire_single_step + 1, :]).astype('int64')
            if acq_selec_method == "A":
                selec_index = np.random.choice(range(65 - (acquire_single_step + 1) * num_selec_point_from_pool),
                                               num_selec_point_from_pool, replace=False)
                acq_index_update = selec_index
            else:
                selec_index = selection(tds_select, model_dir_goes_into_act_stage,
                                        [acq_selec_method], acq_index_rm,
                                        num_selec_point_from_pool, agg_method, agg_quantile_cri,
                                        data_path=training_data_path)
                acq_index_update = selec_index[:, 0]
            print(acquire_single_step, acq_index_update, np.shape(acq_index_update))
            # np.save(os.path.join(model_dir, 'acqu_index'), Acq_Index_Update)
            np.save(os.path.join(logs_path, 'total_select_folder'), total_folder_info)
            np.save(os.path.join(logs_path, 'total_acqu_index'), acq_index_old)


def train_full(resnet_ckpt, acq_method, acq_index_old, acq_index_update, ckpt_dir, model_dir, epoch_size, decay_steps,
               epsilon_opt, batch_size, using_full_training_data=False, flag_pretrain=False):
    # --------Here lots of parameters need to be set------Or maybe we could set it in the configuration file-----#
    # batch_size = 5
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    image_w, image_h, image_c = [480, 480, 3]
    image_shape = np.array([image_w, image_h, image_c])
    targ_height_npy = 528  # this is for padding images
    targ_width_npy = 784  # this is for padding images
    flag_decay = True
    if (acq_method == "F") and (acq_index_old is None):
        learning_rate = 0.0009
    else:
        learning_rate = 0.001
    decay_rate = 0.1
    save_checkpoint_period = 200
    # epsilon_opt = 0.001
    flag_l2_regu = True
    ckpt_dir = ckpt_dir
    moving_average_decay = 0.999

    auxi_weight_num = 1
    auxi_decay_step = 300
    val_step_size = 10
    selec_training_index = np.zeros([2, 5])
    selec_training_index[0, :] = [0, 1, 2, 3, 4]  # this is the index for the initial benign images
    selec_training_index[1, :] = [2, 4, 5, 6, 7]  # this is the index for the initial malignant images
    selec_training_index = selec_training_index.astype('int64')

    checkpoint_path = os.path.join(model_dir, 'model.ckpt')

    with tf.Graph().as_default():
        #  This three placeholder is for extracting the augmented training data##
        image_aug_placeholder = tf.placeholder(tf.float32, [batch_size, targ_height_npy, targ_width_npy, 3])
        label_aug_placeholder = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy, 1])
        edge_aug_placeholder = tf.placeholder(tf.int64, [batch_size, targ_height_npy, targ_width_npy, 1])
        # The placeholder below is for extracting the input for the network #####
        images_train = tf.placeholder(tf.float32, [batch_size, image_w, image_h, image_c])
        instance_labels_train = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        edges_labels_train = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        phase_train = tf.placeholder(tf.bool, shape=None, name="training_state")
        dropout_phase = tf.placeholder(tf.bool, shape=None, name="dropout_state")
        auxi_weight = tf.placeholder(tf.float32, shape=None, name="auxiliary_weight")
        global_step = tf.train.get_or_create_global_step()
        #  -----------------Here is for preparing the dataset for training, pooling and validation---------#
        data_train, data_pool, data_val = prepare_train_data(training_data_path, selec_training_index[0, :],
                                                             selec_training_index[1, :])
        x_image_tr, y_label_tr, y_edge_tr, y_imindex_tr, y_clsindex_tr = data_train
        x_image_pl, y_label_pl, y_edge_pl, y_imindex_pl, y_clsindex_pl = data_pool
        x_image_val, y_label_val, y_edge_val, y_imindex_val, y_clsindex_val = data_val
        y_imindex_pl = np.array(y_imindex_pl)
        y_clsindex_pl = np.array(y_clsindex_pl)

        im_group = [[x_image_tr, y_label_tr, y_edge_tr],
                    [x_image_pl, y_label_pl, y_edge_pl],
                    [x_image_val, y_label_val, y_edge_val]]

        for iterr, single_im_group in enumerate(im_group):
            single_group_new = padding_training_data(single_im_group[0], single_im_group[1], single_im_group[2],
                                                     targ_height_npy, targ_width_npy)
            im_group[iterr] = single_group_new
        x_tr_group = [im_group[0][0], im_group[0][1], im_group[0][2], y_imindex_tr, y_clsindex_tr]
        x_pl_group = [im_group[1][0], im_group[1][1], im_group[1][2], y_imindex_pl, y_clsindex_pl]
        x_image_val, y_label_val, y_edge_val = im_group[2][0], im_group[2][1], im_group[2][2]
        print("-----Before updating, the shape for the training data and pool data-----")
        [print(np.shape(v), np.shape(q)) for v, q in zip(x_tr_group, x_pl_group)]
        if acq_index_old is not None:
            for remove_data in range(np.shape(acq_index_old)[0]):
                num_images_in_pool = range(np.shape(x_pl_group[0])[0])
                images_index_left = np.delete(num_images_in_pool, acq_index_old[remove_data, :])
                image_index_add_to_tr = acq_index_old[remove_data, :]
                print("At step %d" % remove_data, "the index that needs remove", image_index_add_to_tr,
                      "the images that are left in the pool set", images_index_left)
                for i in range(5):
                    x_tr_group[i] = np.concatenate([x_tr_group[i], x_pl_group[i][image_index_add_to_tr]],
                                                   axis=0)
                    x_pl_group[i] = x_pl_group[i][images_index_left]
                print("the removed images' index", acq_index_old[remove_data, :])
                print("there are %d training images and %d pool images after %d step" % (np.shape(x_tr_group[0])[0],
                                                                                         np.shape(x_pl_group[0])[0],
                                                                                         remove_data + 1))
        if acq_index_update is not None:
            for i in range(5):
                x_tr_group[i] = np.concatenate([x_tr_group[i], x_pl_group[i][acq_index_update]],
                                               axis=0)
            print("there are %d training images " % np.shape(x_tr_group[0])[0])
            print([np.shape(v) for v in x_tr_group])
        if using_full_training_data is True:
            for i in range(5):
                x_tr_group[i] = np.concatenate([x_tr_group[i], x_pl_group[i]], axis=0)
        print("---After updating, the shape for the training data and pool data-------")
        [print(np.shape(v), np.shape(q)) for v, q in zip(x_tr_group, x_pl_group)]
        x_image_tr, y_label_tr, y_edge_tr, y_imindex_tr, y_clsindex_tr = x_tr_group

        iteration = np.shape(x_image_tr)[0] // batch_size
        bi_mask_tr = tf.constant(0, shape=[batch_size, targ_height_npy, targ_width_npy, 1],
                                 dtype=tf.int64)
        bi_mask_val = tf.constant(0, shape=[batch_size, targ_height_npy, targ_width_npy, 1],
                                  dtype=tf.int64)

        # ----------Perform data augmentation only on training data-----------------------------------------#
        x_image_aug, y_label_aug, y_edge_aug, _ = aug_train_data(image_aug_placeholder, label_aug_placeholder,
                                                                 edge_aug_placeholder, bi_mask_tr,
                                                                 batch_size, True, image_shape)
        x_image_aug_val, y_label_aug_val, y_edge_aug_val, _ = aug_train_data(image_aug_placeholder,
                                                                             label_aug_placeholder,
                                                                             edge_aug_placeholder, bi_mask_val,
                                                                             batch_size, True,
                                                                             image_shape)

        # ------------------------------Here is for build up the network----------------------------------#
        fb_logits, ed_logits = ResNet_V2_DMNN(images=images_train, training_state=phase_train,
                                              dropout_state=dropout_phase, Num_Classes=2)

        edge_loss, ed_accu, ed_auc_score = Loss(logits=ed_logits, labels=edges_labels_train,
                                                binary_mask=tf.ones([batch_size, image_h, image_w,
                                                                     1], dtype=tf.float32),
                                                auxi_weight=auxi_weight, loss_name="ed")
        fb_loss, fb_accu, fb_auc_score = Loss(logits=fb_logits, labels=instance_labels_train,
                                              binary_mask=tf.ones([batch_size, image_h, image_w,
                                                                   1], dtype=tf.float32),
                                              auxi_weight=auxi_weight, loss_name="fg")

        var_train = tf.trainable_variables()
        total_loss = edge_loss + fb_loss
        if flag_l2_regu is True:
            var_l2 = [v for v in var_train if (('kernel' in v.name) or ('weights' in v.name))]
            total_loss = tf.add_n(
                [total_loss, tf.add_n([tf.nn.l2_loss(v) for v in var_l2 if 'logits' not in v.name]) * 0.001],
                name="Total_Loss")
        # var_opt = [v for v in var_train if ('resnet' not in v.name)]

        train = train_op_batchnorm(total_loss=total_loss, global_step=global_step, initial_learning_rate=learning_rate,
                                   lr_decay_rate=decay_rate, decay_steps=decay_steps,
                                   epsilon_opt=epsilon_opt, var_opt=tf.trainable_variables(),
                                   MOVING_AVERAGE_DECAY=moving_average_decay)

        # summary_op = tf.summary.merge_all()
        if flag_pretrain is False:
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
        print("Use the Learning rate weight decay", flag_decay)
        print("The learning is decayed every %d steps by %.3f " % (decay_steps, decay_rate))
        print("The moving average parameter is ", moving_average_decay)
        print("Batch Size:", batch_size)
        print("Max epochs: ", epoch_size)
        print("Use pretrained model:", flag_pretrain)
        print("The checkpoing file is saved every %d steps" % save_checkpoint_period)
        print("The L2 regularization is turned on:", flag_l2_regu)
        print(" =====================================================")
        with tf.Session() as sess:
            if flag_pretrain is False:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver_set_resnet.restore(sess, resnet_ckpt)
            else:
                ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver_set_all.restore(sess, ckpt.model_checkpoint_path)
                    print("restore parameter from ", ckpt.model_checkpoint_path)
            all_files = os.listdir(model_dir)
            for v in all_files:
                os.remove(os.path.join(model_dir, v))
                print("----------removing initial trained files-------------------", v)
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
                x_image_sh, y_label_sh, y_edge_sh, y_imindex_sh, y_clsindex_sh = shuffle(x_image_tr, y_label_tr,
                                                                                         y_edge_tr, y_imindex_tr,
                                                                                         y_clsindex_tr)

                batch_index = 0

                train_stat_per_epoch = np.zeros([iteration, 4])
                for single_batch in range(iteration):
                    x_image_batch, y_label_batch, y_edge_batch, \
                    _, batch_index = generate_batch(x_image_sh, y_label_sh, y_edge_sh,
                                                    np.ones([len(x_image_tr), targ_height_npy,
                                                             targ_width_npy, 1]), batch_index, batch_size)
                    feed_dict_aug = {image_aug_placeholder: x_image_batch, label_aug_placeholder: y_label_batch,
                                     edge_aug_placeholder: y_edge_batch}
                    x_image_npy, y_label_npy, y_edge_npy = sess.run([x_image_aug, y_label_aug, y_edge_aug],
                                                                    feed_dict=feed_dict_aug)

                    feed_dict_op = {images_train: x_image_npy,
                                    instance_labels_train: y_label_npy,
                                    edges_labels_train: y_edge_npy,
                                    auxi_weight: auxi_weight_num,
                                    phase_train: True,
                                    dropout_phase: True}
                    fetches_train = [train, fb_loss, fb_accu, fb_auc_score, edge_loss]
                    _, _fb_loss, _fb_f1, _fb_auc, _ed_loss = sess.run(fetches=fetches_train, feed_dict=feed_dict_op)
                    train_stat_per_epoch[single_batch, 0] = _fb_loss
                    train_stat_per_epoch[single_batch, 1] = _fb_f1
                    train_stat_per_epoch[single_batch, 2] = _fb_auc
                    train_stat_per_epoch[single_batch, 3] = _ed_loss
                train_tot_stat[single_epoch, :] = np.mean(train_stat_per_epoch, axis=0)
                print(single_epoch, train_tot_stat[single_epoch, :])

                if single_epoch % val_step_size == 0:
                    val_iteration = np.shape(x_image_val)[0] // batch_size
                    print("start validating .......with %d images and %d iterations" % (np.shape(x_image_val)[0],
                                                                                        val_iteration))

                    val_batch_index = 0
                    val_stat_per_epoch = np.zeros([val_iteration, 4])
                    for single_batch_val in range(val_iteration):
                        x_image_batch_val, y_label_batch_val, y_edge_batch_val, _, val_batch_index = generate_batch(
                            x_image_val, y_label_val, y_edge_val,
                            np.ones([len(x_image_val), targ_height_npy, targ_width_npy, 1]),
                            val_batch_index, batch_size)
                        feed_dict_aug_val = {image_aug_placeholder: x_image_batch_val,
                                             label_aug_placeholder: y_label_batch_val,
                                             edge_aug_placeholder: y_edge_batch_val}
                        x_image_npy_val, y_label_npy_val, y_edge_npy_val = sess.run(
                            [x_image_aug_val, y_label_aug_val, y_edge_aug_val], feed_dict=feed_dict_aug_val)

                        fetches_valid = [fb_loss, fb_accu, fb_auc_score, edge_loss]
                        feed_dict_valid = {images_train: x_image_npy_val,
                                           instance_labels_train: y_label_npy_val,
                                           edges_labels_train: y_edge_npy_val,
                                           auxi_weight: 0,
                                           phase_train: False,
                                           dropout_phase: False}
                        _fbloss_val, _fb_f1_val, _fb_auc_val, _edloss_val = sess.run(fetches=fetches_valid,
                                                                                     feed_dict=feed_dict_valid)
                        val_stat_per_epoch[single_batch_val, 0] = _fbloss_val
                        val_stat_per_epoch[single_batch_val, 1] = _fb_f1_val
                        val_stat_per_epoch[single_batch_val, 2] = _fb_auc_val
                        val_stat_per_epoch[single_batch_val, 3] = _edloss_val

                    val_tot_stat[single_epoch // val_step_size, :] = np.mean(val_stat_per_epoch, axis=0)
                    print("validation", single_epoch, val_tot_stat[single_epoch // val_step_size, :])

                if single_epoch % save_checkpoint_period == 0 or single_epoch == (epoch_size - 1):
                    saver_set_all.save(sess, checkpoint_path, global_step=single_epoch)
                if single_epoch == (epoch_size - 1):
                    print("Acq Index Update", acq_index_update)
                    np.save(os.path.join(model_dir, 'trainstat'), train_tot_stat)
                    np.save(os.path.join(model_dir, 'valstat'), val_tot_stat)

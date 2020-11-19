# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:56:26 2018

@author: s161488
"""
############################################################
#  Loss Functions
############################################################
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc


def calc_f1_score(pred, label):
    """
    pred: [batch_size, im_h, im_w, 1]
    label: [batch_size, im_h, im_w, 1]
    """
    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    f1score = f1_score(y_true=label, y_pred=pred)

    return f1score.astype('float32')


def tf_f1score(pred, label):
    f1score_tensor = tf.py_func(calc_f1_score, [pred, label], tf.float32)
    return f1score_tensor


def calc_auc_score(pred, label):
    """
    In this function, the pred is not just predicted label, instead it's the predicted probability
    pred: [batch_size, im_h, im_w, 1]
    label: [batch_size, im_h, im_w, 1]
    """
    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    fpr, tpr, thresholds = roc_curve(y_true=label, y_score=pred,
                                     pos_label=1)
    auc_value = auc(fpr, tpr)
    return auc_value.astype('float32')


def tf_auc_score(pred, label):
    auc_score_tensor = tf.py_func(calc_auc_score, [pred, label], tf.float32)
    return auc_score_tensor


def Loss(logits, labels, binary_mask, auxi_weight, loss_name):
    """Calculate the cross entropy loss for the edge detection.
    Args:
    logits: The logits output from the edge detection channel. Shape [6, Num_Batch, Image_Height, Image_Width, 1] (6 is because 5 
    side-output, and 1 fuse-output). 
    labels: The edge label. Shape [Num_Batch, Image_height, Image_Width, 1].
    DCAN: A boolean variable. If DCAN is True. The last channel we use softmax_cross_entropy. If it's False, then last feature map
    we use sigmoid_loss. 
    
    Returns:
    Edge_loss: the total loss for the edge detection channel.
    
    Operations:
    Because there is a class imbalancing situation, the pixels belong to background must be much larger than the pixels belong to
    edge, so that we add a penalty beta. beta = Y_/Y. 
    """
    y = tf.reshape(labels, [-1])
    y = tf.cast(tf.not_equal(y, 0), tf.int32)
    Num_Map = np.shape(logits)[0]
    cost = 0
    binary_mask = tf.reshape(binary_mask, [-1])  # [batch_size, imh, imw, 1]
    binary_mask = tf.cast(tf.not_equal(binary_mask, 0), tf.int32)
    for i in range(Num_Map):
        cross_entropy_sep = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits[i], [-1, 2]),
                                                                           labels=y)
        cross_entropy_sep = tf.boolean_mask(cross_entropy_sep, tf.equal(binary_mask, 1))
        cross_entropy_sep = tf.reduce_mean(cross_entropy_sep, name='auxiliary' + loss_name + '%d' % i)
        tf.add_to_collection('loss', cross_entropy_sep)
        cost += cross_entropy_sep
    fuse_map = tf.add_n(logits)
    fuse_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=tf.reshape(fuse_map, [-1, 2]))
    fuse_cost = tf.reduce_mean(tf.boolean_mask(fuse_cost, tf.equal(binary_mask, 1)), name=loss_name + 'fuse_cost')
    tf.add_to_collection('loss', fuse_cost)
    cost = auxi_weight * cost + fuse_cost
    pred = tf.reshape(tf.argmax(fuse_map, axis=-1, output_type=tf.int32), [-1])
    pred_bi = tf.boolean_mask(pred, tf.equal(binary_mask, 1))
    y_bi = tf.boolean_mask(y, tf.equal(binary_mask, 1))
    pred_bi_cond_f1 = tf.equal(tf.reduce_sum(pred_bi), 0)
    y_bi_cond_f1 = tf.equal(tf.reduce_sum(y_bi), 0)
    accuracy = tf.cond(tf.logical_and(pred_bi_cond_f1, y_bi_cond_f1),
                       lambda: tf.constant(1.0),
                       lambda: tf_f1score(pred=pred_bi, label=y_bi))
    auc_pred_bi = tf.boolean_mask(tf.reshape(fuse_map[:, :, :, 1], [-1]), tf.equal(binary_mask, 1))
    auc_score = tf.cond(tf.equal(tf.reduce_mean(y_bi), 1),
                        lambda: tf.constant(0.0),
                        lambda: tf_auc_score(pred=auc_pred_bi, label=y_bi))
    return cost, accuracy, auc_score


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('loss')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    return loss_averages_op


def train_op_batchnorm(total_loss, global_step, initial_learning_rate, lr_decay_rate, decay_steps, epsilon_opt, var_opt,
                       MOVING_AVERAGE_DECAY):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    lr_decay_rate,
                                    staircase=False)
    # if staircase is True, then the division between global_step/decay_steps is a interger,
    # otherwise it's not an interger.
    tf.summary.scalar('learning_rate', lr)

    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variables_averages.apply(var_opt)
    with tf.control_dependencies(update_ops):
        loss_averages_op = _add_loss_summaries(total_loss)
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr, epsilon=epsilon_opt)
            grads = opt.compute_gradients(total_loss, var_list=var_opt)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = variables_averages.apply(var_opt)

    return train_op

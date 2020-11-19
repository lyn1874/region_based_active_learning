# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:41:21 2018
This files includes all the encoder process, the possible encoder process including VGG16, AlexNet, and ResNet. Then the output 
from the encoder are utilized as the initialization for different decoder channels.
@author: s161488
"""
import tensorflow as tf
from models.layer_utils import conv_layer_renew, deconv_layer_renew
import models.resnet_v2 as resnet_v2
import os
import tensorflow.contrib.slim as slim


def ResNet_V2_DMNN(images, training_state, dropout_state, Num_Classes):
    arch_name = "resnet_v2_50"
    images = tf.image.random_brightness(images, max_delta=10.0)
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        _, end_points = resnet_v2.resnet_v2_50(images, num_classes=Num_Classes, dropout_phase=dropout_state,
                                               is_training=training_state, global_pool=False, output_stride=16,
                                               spatial_squeeze=False)

    deconv_u1 = deconv_layer_renew(end_points[os.path.join(arch_name, 'block4')], filter_shape=24,
                                   output_channel=Num_Classes, name="deconv_layer_u1", strides=16,
                                   training_state=training_state)
    aux_conv_c1 = conv_layer_renew(deconv_u1, "aux_conv_c1", [1, Num_Classes], training_state=training_state)
    deconv_u2 = deconv_layer_renew(end_points[os.path.join(arch_name, 'block3')], filter_shape=24,
                                   output_channel=Num_Classes, name="deconv_layer_u2", strides=16,
                                   training_state=training_state)
    aux_conv_c2 = conv_layer_renew(deconv_u2, "aux_conv_c2", [1, Num_Classes], training_state=training_state)
    deconv_u3 = deconv_layer_renew(end_points[os.path.join(arch_name, 'block2')], filter_shape=24,
                                   output_channel=Num_Classes, name="deconv_layer_u3", strides=16,
                                   training_state=training_state)
    aux_conv_c3 = conv_layer_renew(deconv_u3, "aux_conv_c3", [1, Num_Classes], training_state=training_state)

    deconv_u1_edge = deconv_layer_renew(end_points[os.path.join(arch_name, 'block4')], filter_shape=24,
                                        output_channel=Num_Classes, name="deconv_layer_u1_edge", strides=16,
                                        training_state=training_state)
    aux_conv_c1_edge = conv_layer_renew(deconv_u1_edge, "aux_conv_c1_edge", [1, Num_Classes],
                                        training_state=training_state)
    deconv_u2_edge = deconv_layer_renew(end_points[os.path.join(arch_name, 'block3')], filter_shape=24,
                                        output_channel=Num_Classes, name="deconv_layer_u2_edge", strides=16,
                                        training_state=training_state)
    aux_conv_c2_edge = conv_layer_renew(deconv_u2_edge, "aux_conv_c2_edge", [1, Num_Classes],
                                        training_state=training_state)
    deconv_u3_edge = deconv_layer_renew(end_points[os.path.join(arch_name, 'block2')], filter_shape=24,
                                        output_channel=Num_Classes, name="deconv_layer_u3_edge", strides=16,
                                        training_state=training_state)
    aux_conv_c3_edge = conv_layer_renew(deconv_u3_edge, "aux_conv_c3_edge", [1, Num_Classes],
                                        training_state=training_state)
    fb_pred_logit = [aux_conv_c3, aux_conv_c2, aux_conv_c1]
    ed_pred_logit = [aux_conv_c3_edge, aux_conv_c2_edge, aux_conv_c1_edge]

    return fb_pred_logit, ed_pred_logit

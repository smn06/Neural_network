import tensorflow as tf
import numpy as np


slim = tf.contrib.slim


def conv(inputs, out_channels, kernel_size=3, stride=1):
    if stride > 1:
        inputs = pad(inputs, kernel_size)

    
    outputs = slim.conv2d(inputs, out_channels, kernel_size, stride=stride,
                          pad=('SAME' if stride == 1 else 'VALID'))
    return outputs


def pad(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start
    outputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]])
    return outputs



def block(inputs, channels, res_num):
    net = conv(inputs, channels * 2, stride=2)
    route = conv(net, channels, kernel_size=1)
    net = conv(net, channels, kernel_size=1)

    for _ in range(res_num):
        tmp = net
        net = conv(net, channels, kernel_size=1)
        net = conv(net, channels)
        net = tmp + net

    net = conv(net, channels, kernel_size=1)
    net = tf.concat([net, route], -1)
    net = conv(net, channels * 2, kernel_size=1)

    return net



def conv(inputs, channels, a, b):
    net = inputs
    for _ in range(a):
        net = conv(net, channels / 2, kernel_size=1)
        net = conv(net, channels)

    for _ in range(b):
        channels /= 2
        net = conv(net, channels, kernel_size=1)
    outputs = net
    return outputs



def max_block(inputs):
    maxpool_5 = tf.nn.max_pool(inputs, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
    maxpool_9 = tf.nn.max_pool(inputs, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
    maxpool_13 = tf.nn.max_pool(inputs, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
    outputs = tf.concat([maxpool_13, maxpool_9, maxpool_5, inputs], -1)
    return outputs



def yolo_upsample_block(inputs, in_channels, route):
    shape = tf.shape(inputs)
    out_height, out_width = shape[1] * 2, shape[2] * 2
    inputs = tf.image.resize_nearest_neighbor(inputs, (out_height, out_width))

    route = conv(route, in_channels, kernel_size=1)

    outputs = tf.concat([route, inputs], -1)
    return outputs


def mish(inputs):
    threshold = 20.0
    tmp = inputs
    inputs = tf.where(tf.math.logical_and(tf.less(inputs, threshold), tf.greater(inputs, -threshold)),
                      tf.log(1 + tf.exp(inputs)),
                      tf.zeros_like(inputs))
    inputs = tf.where(tf.less(inputs, -threshold),
                      tf.exp(inputs),
                      inputs)
    outputs = tmp * tf.tanh(inputs)
    return outputs


class y:
    def __init__(self, classes, vals):
        self.classes = classes
        self.vals = vals
        self.width = 608
        self.height = 608
        pass

    def inf(self, inputs, batch_norm_decay=0.9, weight_decay=0.0005, istraining=True, reuse=False):
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': istraining,
            'fused': None,  
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                
                with slim.arg_scope([slim.conv2d], activation_fn=mish):
                    with tf.variable_scope('Downsample'):
                        net = conv(inputs, 32)
                        
                        net = conv(net, 64, stride=2)
                        
                        route = conv(net, 64, kernel_size=1)
                        net = conv(net, 64, kernel_size=1)
                        tmp = net
                        net = conv(net, 32, kernel_size=1)
                        net = conv(net, 64)
                        net = tmp + net
                        net = conv(net, 64, kernel_size=1)
                        net = tf.concat([net, route], -1)
                        
                        net = conv(net, 64, kernel_size=1)

                        
                        net = block(net, 64, 2)
                        
                        net = block(net, 128, 8)
                        
                        up_route_54 = net

                        net = block(net, 256, 8)
                        
                        up_route_85 = net

                        net = block(net, 512, 4)

                
                with slim.arg_scope([slim.conv2d], activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                    with tf.variable_scope('leaky_relu'):
                        
                        net = conv(net, 1024, 1, 1)
                        
                        net = max_block(net)
                        
                        net = conv(net, 512, kernel_size=1)
                        net = conv(net, 1024)
                        net = conv(net, 512, kernel_size=1)
                        
                        route_3 = net

                        net = conv(net, 256, kernel_size=1)
                        
                        net = yolo_upsample_block(net, 256, up_route_85)
                        
                        net = conv(net, 512, 2, 1)
                        
                        route_2 = net

                        net = conv(net, 128, kernel_size=1)
                        
                        net = yolo_upsample_block(net, 128, up_route_54)
                        
                        net = conv(net, 256, 2, 1)
                        
                        route_1 = net

                    with tf.variable_scope('yolo'):
                        net = conv(route_1, 256)
                        
                        net = slim.conv2d(net, 3 * (4 + 1 + self.classes), 1,
                                          stride=1, normalizer_fn=None, activation_fn=None,
                                          biases_initializer=tf.zeros_initializer())
                        
                        feature_y3 = net

                        net = conv(route_1, 256, stride=2)
                        
                        net = tf.concat([net, route_2], -1)
                        
                        net = conv(net, 512, 2, 1)
                        
                        route_147 = net

                        net = conv(net, 512)
                        
                        net = slim.conv2d(net, 3*(4+1+self.classes), 1,
                                          stride=1, normalizer_fn=None, activation_fn=None,
                                          biases_initializer=tf.zeros_initializer())
                        
                        feature_y2 = net

                        net = conv(route_147, 512, stride=2)
                        
                        net = tf.concat([net, route_3], -1)
                        
                        net = conv(net, 1024, 3, 0)
                        
                        net = slim.conv2d(net, 3*(4+1+self.classes), 1,
                                          stride=1, normalizer_fn=None,
                                          activation_fn=None, biases_initializer=tf.zeros_initializer())
                        
                        feature_y1 = net
        if not istraining:
            return self.predict(feature_y1, feature_y2, feature_y3)
        return feature_y1, feature_y2, feature_y3

    

    def iou(pre_xy, pre_wh, valid_yi_true):

        
        pre_xy = tf.expand_dims(pre_xy, -2)
        pre_wh = tf.expand_dims(pre_wh, -2)

        
        yi_true_xy = valid_yi_true[..., 0:2]
        yi_true_wh = valid_yi_true[..., 2:4]

        
        intersection_left_top = tf.maximum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        
        intersection_right_bottom = tf.minimum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))
        
        combine_left_top = tf.minimum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        
        combine_right_bottom = tf.maximum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))

        
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)
        
        
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
        
        pre_area = pre_wh[..., 0] * pre_wh[..., 1]
        
        true_area = yi_true_wh[..., 0] * yi_true_wh[..., 1]
        
        true_area = tf.expand_dims(true_area, axis=0)
        
        iou = intersection_area / (pre_area + true_area - intersection_area + 1e-10)  

        
        combine_area = combine_wh[..., 0] * combine_wh[..., 1]
        
        giou = (intersection_area+1e-10) / combine_area  
        
        return iou, giou

    
    @staticmethod
    def __get_ciou_loss(pre_xy, pre_wh, yi_box):

        
        yi_true_xy = yi_box[..., 0:2]
        yi_true_wh = yi_box[..., 2:4]

        
        pre_lt = pre_xy - pre_wh/2
        pre_rb = pre_xy + pre_wh/2
        truth_lt = yi_true_xy - yi_true_wh / 2
        truth_rb = yi_true_xy + yi_true_wh / 2

        
        intersection_left_top = tf.maximum(pre_lt, truth_lt)
        intersection_right_bottom = tf.minimum(pre_rb, truth_rb)
        
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        
        intersection_area = intersection_wh[..., 0:1] * intersection_wh[..., 1:2]
        
        combine_left_top = tf.minimum(pre_lt, truth_lt)
        
        combine_right_bottom = tf.maximum(pre_rb, truth_rb)
        
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)

        
        C = tf.square(combine_wh[..., 0:1]) + tf.square(combine_wh[..., 1:2])
        
        D = tf.square(yi_true_xy[..., 0:1] - pre_xy[..., 0:1]) + tf.square(yi_true_xy[..., 1:2] - pre_xy[..., 1:2])

        
        pre_area = pre_wh[..., 0:1] * pre_wh[..., 1:2]
        true_area = yi_true_wh[..., 0:1] * yi_true_wh[..., 1:2]

        
        iou = intersection_area / (pre_area + true_area - intersection_area)

        pi = 3.14159265358979323846

        
        v = 4 / (pi * pi) * tf.square(tf.subtract(tf.math.atan(yi_true_wh[..., 0:1] / yi_true_wh[..., 1:2]), tf.math.atan(pre_wh[..., 0:1] / pre_wh[..., 1:2])))

        
        
        alpha = v / (1.0 - iou + v)
        ciou_loss = 1.0 - iou + D / C + alpha * v
        return ciou_loss

    
    def __get_low_iou_mask(self, pre_xy, pre_wh, yi_true, use_iou=True, ignore_thresh=0.5):

        conf_yi_true = yi_true[..., 4:5]

        
        low_iou_mask = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        
        def loop_cond(index):
            
            return tf.less(index, tf.shape(yi_true)[0])

        def loop_body(index, mask):
            
            valid_yi_true = tf.boolean_mask(yi_true[index], tf.cast(conf_yi_true[index, ..., 0], tf.bool))
            
            iou, giou = self.iou(pre_xy[index], pre_wh[index], valid_yi_true)

            
            if use_iou:
                best_giou = tf.reduce_max(iou, axis=-1)
            else:
                best_giou = tf.reduce_max(giou, axis=-1)
            
            low_iou_mask_tmp = best_giou < ignore_thresh
            
            low_iou_mask_tmp = tf.expand_dims(low_iou_mask_tmp, -1)
            
            mask = mask.write(index, low_iou_mask_tmp)
            return index + 1, mask

        _, low_iou_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, low_iou_mask])
        
        low_iou_mask = low_iou_mask.stack()
        return low_iou_mask

    
    @staticmethod
    def __get_low_prob_mask(prob, prob_thresh=0.25):

        max_prob = tf.reduce_max(prob, axis=-1, keepdims=True)
        low_prob_mask = max_prob < prob_thresh        
        return low_prob_mask

    
    def __decode_feature(self, yi_pred, curr_anchors):

        shape = tf.cast(tf.shape(yi_pred), tf.float32)
        batch_size, grid_size = shape[0], shape[1]
        
        yi_pred = tf.reshape(yi_pred, [batch_size, grid_size, grid_size, 3, 5 + self.classes])
        
        
        
        
        xy, wh, conf, prob = tf.split(yi_pred, [2, 2, 1, self.classes], axis=-1)

        offset_x = tf.range(grid_size, dtype=tf.float32)  
        offset_y = tf.range(grid_size, dtype=tf.float32)  
        offset_x, offset_y = tf.meshgrid(offset_x, offset_y)
        offset_x = tf.reshape(offset_x, (-1, 1))
        offset_y = tf.reshape(offset_y, (-1, 1))
        
        offset_xy = tf.concat([offset_x, offset_y], axis=-1)
        
        offset_xy = tf.reshape(offset_xy, [grid_size, grid_size, 1, 2])
        
        xy = tf.math.sigmoid(xy) + offset_xy
        xy = xy / [grid_size, grid_size]

        wh = tf.math.exp(wh) * curr_anchors
        wh = wh / [self.width, self.height]

        return xy, wh, conf, prob

    
    def __compute_loss_v4(self, xy, wh, conf, prob, yi_true, cls_normalizer=1.0, ignore_thresh=0.5, prob_thresh=0.25, score_thresh=0.25, iou_normalizer=0.07):

        low_iou_mask = self.__get_low_iou_mask(xy, wh, yi_true, ignore_thresh=ignore_thresh)
        
        low_prob_mask = self.__get_low_prob_mask(prob, prob_thresh=prob_thresh)        
        
        low_iou_prob_mask = tf.math.logical_or(low_iou_mask, low_prob_mask)
        low_iou_prob_mask = tf.cast(low_iou_prob_mask, tf.float32)

        batch_size = tf.cast(tf.shape(xy)[0], tf.float32)

        
        conf_scale = wh[..., 0:1] * wh[..., 1:2]
        conf_scale = tf.where(tf.math.greater(conf_scale, 0), tf.math.sqrt(conf_scale), conf_scale)
        conf_scale = conf_scale * cls_normalizer                                                        
        conf_scale = tf.math.square(conf_scale)

        
        no_obj_mask = 1.0 - yi_true[..., 4:5]
        conf_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[:, :, :, :, 4:5], logits=conf) * conf_scale * no_obj_mask * low_iou_prob_mask

        
        obj_mask = yi_true[..., 4:5]
        conf_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[:, :, :, :, 4:5], logits=conf) * np.square(cls_normalizer) * obj_mask

        
        conf_loss = conf_loss_obj + conf_loss_no_obj
        conf_loss = tf.clip_by_value(conf_loss, 0.0, 1e3)
        conf_loss = tf.reduce_sum(conf_loss) / batch_size
        conf_loss = tf.clip_by_value(conf_loss, 0.0, 1e4)

        
        yi_true_ciou = tf.where(tf.math.less(yi_true[..., 0:4], 1e-10), tf.ones_like(yi_true[..., 0:4]), yi_true[..., 0:4])
        pre_xy = tf.where(tf.math.less(xy, 1e-10), tf.ones_like(xy), xy)
        pre_wh = tf.where(tf.math.less(wh, 1e-10), tf.ones_like(wh), wh)
        ciou_loss = self.__get_ciou_loss(pre_xy, pre_wh, yi_true_ciou)
        ciou_loss = tf.where(tf.math.greater(obj_mask, 0.5), ciou_loss, tf.zeros_like(ciou_loss))
        ciou_loss = tf.square(ciou_loss * obj_mask) * iou_normalizer
        ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e3)
        ciou_loss = tf.reduce_sum(ciou_loss) / batch_size
        ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e4)

        
        xy = tf.clip_by_value(xy, 1e-10, 1e4)
        xy_loss = obj_mask * tf.square(yi_true[..., 0: 2] - xy)
        xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e3)
        xy_loss = tf.reduce_sum(xy_loss) / batch_size
        xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e4)

        
        wh_y_true = tf.where(condition=tf.math.less(yi_true[..., 2:4], 1e-10), x=tf.ones_like(yi_true[..., 2: 4]), y=yi_true[..., 2: 4])
        wh_y_pred = tf.where(condition=tf.math.less(wh, 1e-10), x=tf.ones_like(wh), y=wh)
        wh_y_true = tf.clip_by_value(wh_y_true, 1e-10, 1e10)
        wh_y_pred = tf.clip_by_value(wh_y_pred, 1e-10, 1e10)
        wh_y_true = tf.math.log(wh_y_true)
        wh_y_pred = tf.math.log(wh_y_pred)

        wh_loss = obj_mask * tf.square(wh_y_true - wh_y_pred)
        wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e3)
        wh_loss = tf.reduce_sum(wh_loss) / batch_size
        wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e4)
        
        
        score = prob * conf
        
        high_score_mask = score > score_thresh
        high_score_mask = tf.cast(high_score_mask, tf.float32)
        
        class_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[..., 5:5+self.classes],
                                                        logits=prob 
                                                    ) * low_iou_prob_mask * no_obj_mask * high_score_mask
        
        class_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[..., 5:5+self.classes],
                                                        logits=prob
                                                    ) * obj_mask

        class_loss = class_loss_no_obj + class_loss_obj        
        class_loss = tf.clip_by_value(class_loss, 0.0, 1e3)
        class_loss = tf.reduce_sum(class_loss) / batch_size
        class_loss = tf.clip_by_value(class_loss, 0.0, 1e4)

        loss_total = xy_loss + wh_loss + conf_loss + class_loss + ciou_loss
        return loss_total

    
    def loss(self, feature_y1, feature_y2, feature_y3, y1_true, y2_true, y3_true, cls_normalizer=1.0, ignore_thresh=0.5, prob_thresh=0.25, score_thresh=0.25):

        xy, wh, conf, prob = self.__decode_feature(feature_y1, self.vals[2])
        loss_y1 = self.__compute_loss_v4(xy, wh, conf, prob, y1_true, cls_normalizer=cls_normalizer, ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        
        xy, wh, conf, prob = self.__decode_feature(feature_y2, self.vals[1])
        loss_y2 = self.__compute_loss_v4(xy, wh, conf, prob, y2_true, cls_normalizer=cls_normalizer, ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        
        xy, wh, conf, prob = self.__decode_feature(feature_y3, self.vals[0])
        loss_y3 = self.__compute_loss_v4(xy, wh, conf, prob, y3_true, cls_normalizer=cls_normalizer, ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        return loss_y1 + loss_y2 + loss_y3

    
    def box(self, feature, vals):
        xy, wh, conf, prob = self.__decode_feature(feature, vals)
        conf, prob = tf.sigmoid(conf), tf.sigmoid(prob)
        boxes = tf.concat([xy[..., 0: 1] - wh[..., 0: 1] / 2.0,
                           xy[..., 1: 2] - wh[..., 1: 2] / 2.0,
                           xy[..., 0: 1] + wh[..., 0: 1] / 2.0,
                           xy[..., 1: 2] + wh[..., 1: 2] / 2.0], -1)
        shape = tf.shape(feature)
        
        boxes = tf.reshape(boxes, (shape[0], shape[1] * shape[2] * 3, -1))
        
        conf = tf.reshape(conf, (shape[0], shape[1] * shape[2] * 3, 1))
        
        prob = tf.reshape(prob, (shape[0], shape[1] * shape[2] * 3, -1))
        return boxes, conf, prob

    
    def predict(self, feature_y1, feature_y2, feature_y3, score_threshold=0.5, iou_threshold=0.4, max_boxes=100):

        boxes_y1, conf_y1, prob_y1 = self.box(feature_y1, self.vals[2])
        boxes_y2, conf_y2, prob_y2 = self.box(feature_y2, self.vals[1])
        boxes_y3, conf_y3, prob_y3 = self.box(feature_y3, self.vals[0])

        
        boxes = tf.concat([boxes_y1, boxes_y2, boxes_y3], 1)
        
        conf = tf.concat([conf_y1, conf_y2, conf_y3], 1)
        
        prob = tf.concat([prob_y1, prob_y2, prob_y3], 1)
        
        scores = conf * prob

        boxes = tf.reshape(boxes, [-1, 4])
        scores = tf.reshape(scores, [-1, self.classes])

        box_list, score_list = [], []
        for i in range(self.classes):
            nms_indices = tf.image.non_max_suppression(boxes=boxes,
                                                       scores=scores[:, i],
                                                       max_output_size=max_boxes,
                                                       iou_threshold=iou_threshold,
                                                       score_threshold=score_threshold,
                                                       name='nms_indices')

            box_list.append(tf.gather(boxes, nms_indices))
            score_list.append(tf.gather(scores, nms_indices))

        boxes = tf.concat(box_list, axis=0, name='pred_boxes')
        scores = tf.reduce_max(tf.concat(score_list, axis=0), axis=1, name='pred_scores')
        labels = tf.argmax(tf.concat(score_list, axis=0), axis=1, name='pred_labels')

        return boxes, scores, labels

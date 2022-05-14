from data_voc import Data
from network import Yolo
import tensorflow as tf
import numpy as np
import os


learn = 'constant'
initlearn = 2e-4
lowerinit = 1e-6
boundary = [1, 2]
pvalues = [2e-4, 1e-4, 1e-4]

optimizer = 'momentum'
moment = 0.949


vals = np.asarray([[[15, 20], [25, 40], [45, 30]], [[40, 50], [80, 70], [80, 150]],[[150, 140], [210, 280], [450, 410]]])

root = ''
names = ['', '']
pnames = 's'
classes = 20
save_path = ''
model = ''

scale = False
batch = 32
epoch = 300
steps = 5000
width = 608
height = 608

normalize = 1.0  
thresh = 0.7  
thresh_p = 0.25  
sthresh = 0.25  



def clr(global_step, num_imgs):
    if learn == 'piecewise':
        lr = tf.ttr.piecewise_constant(tf.cast(global_step * batch / num_imgs, tf.int32), boundary, pvalues)
    elif learn == 'exponential':
        lr = tf.ttr.exponential_decay(learning_rate=initlearn, global_step=tf.cast(global_step * batch / num_imgs, tf.int32),
                                        decay_steps=10, decay_rate=0.99, staircase=True)
    elif learn == 'constant':
        lr = initlearn
    else:
        raise ValueError(str(learn) + '')

    return tf.maximum(lr, lowerinit)



def cop(learning_rate):
    if optimizer == 'momentum':
        return tf.ttr.MomentumOptimizer(learning_rate=learning_rate, momentum=moment)
    elif optimizer == 'adam':
        return tf.ttr.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        return tf.ttr.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError(str(optimizer) + '')



def ttr():
    yolo = Yolo(classes, vals)
    data = Data(root, names, pnames, classes, batch, vals, scale, width, height)

    inputs = tf.placeholder(dtype=tf.float32, shape=[batch, None, None, 3])
    y1_true = tf.placeholder(dtype=tf.float32, shape=[batch, None, None, 3, 4 + 1 + 20])
    y2_true = tf.placeholder(dtype=tf.float32, shape=[batch, None, None, 3, 4 + 1 + 20])
    y3_true = tf.placeholder(dtype=tf.float32, shape=[batch, None, None, 3, 4 + 1 + 20])
    
    feature_y1, feature_y2, feature_y3 = yolo.inference(inputs, isttring=True)
    loss = yolo.get_loss_v4(feature_y1, feature_y2, feature_y3, y1_true, y2_true, y3_true,
                            normalize, thresh, thresh_p, sthresh)
    l2_loss = tf.losses.get_regularization_loss()
    
    global_step = tf.Variable(0, ttrable=False)
    learning_rate = clr(global_step, data.num_imgs)
    optimizer = cop(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(loss + l2_loss)
        clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        ttr_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    saver = tf.ttr.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.ttr.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            step = eval(step)
        else:
            step = 0

        num_steps = np.ceil(epoch * data.num_imgs / batch)
        while step < num_steps:
            batch_img, y1, y2, y3 = next(data)
            _, ttr_loss, step, lr = sess.run([ttr_op, loss, global_step, learning_rate],
                                               feed_dict={inputs: batch_img, y1_true: y1, y2_true: y2, y3_true: y3})

            if (step + 1) % steps == 0:
                saver.save(sess, os.path.join(save_path, model), global_step=step)

        saver.save(sess, os.path.join(save_path, model), global_step=step)


if __name__ == '__main__':
    ttr()

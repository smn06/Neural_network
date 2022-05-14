from __future__ import division, print_function
from network import y
import tensorflow as tf
import numpy as np

vals = np.asarray([[[15, 20], [25, 40], [45, 30]], [[40, 50], 80, 70], [80, 150]],[[150, 140], [210, 280], [450, 410]])


def weight(path, cpath):
    initt = y(classes=36, vals=vals)
    with tf.Session() as session:
        inputs = tf.placeholder(tf.float32, [1, 608, 608, 3], name='inputs')
        _, _, _ = initt.inference(inputs, istraining=False)
        varl = tf.global_variables()
        saver = tf.train.Saver(varl=varl)

        with open(path, "rb") as fp:
            np.fromfile(fp, dtype=np.int32, count=5)
            weights = np.fromfile(fp, dtype=np.float32)

        p = 0  
        i = 0  
        assign = []  
        while i < len(varl) - 1:
            var1 = varl[i]
            var2 = varl[i + 1]
            
            if 'Conv' in var1.name.split('/')[-2]:

                if 'BatchNorm' in var2.name.split('/')[-2]:
                    gamma, beta, mean, var = varl[i + 1:i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        params = np.prod(shape)
                        
                        vweights = weights[p:p + params].reshape(shape)
                        assign.append(tf.assign(var, vweights, validate_shape=True))
                        p += params
                    
                    i += 4               
                elif 'Conv' in var2.name.split('/')[-2]:
                    shape = var2.shape.as_list()
                    params = np.prod(shape)
                    
                    bweights = weights[p:p + params].reshape(shape)
                    assign.append(tf.assign(var2, bweights, validate_shape=True))
                    p += params
                    
                    i += 1
                shape = var1.shape.as_list()
                params = np.prod(shape)
                vweights = weights[p:p + params].reshape((shape[3], shape[2], shape[0], shape[1]))
                vweights = np.transpose(vweights, (2, 3, 1, 0))
                assign.append(tf.assign(var1, vweights, validate_shape=True))
                p += params
                i += 1
        session.run(assign)
        saver.save(session, save_path=cpath)

if __name__ == '__main__':
    weight('./yolov4.weights', './model')

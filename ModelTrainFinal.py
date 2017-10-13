
# coding: utf-8

# In[1]:

from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import os
import nibabel as nib
from nibabel.testing import data_path


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[74]:

# Data feeding function

def trainDatafeed(batch_size):
    """
    Generate training batches for the BRATS data
    
    The image resizing is done as follows:
        - 40 slices removed from beginning and end of dims 0 and 1
        - 4 removed from start and 7 removed from end of dim 2
        - Image futher downsampled via mean pooling at beginning of CNN
    
    """
    i = 0
    j = 0
    inds = np.random.randint(1, 255, batch_size)
    
    inData = np.zeros((batch_size,160,160,144,4))
    labelData = np.zeros((batch_size,160,160,144,5))
    yFull = np.zeros((batch_size,240,240,155))
    
    for root, dirs, files in os.walk('../HGG/'):
        if (j>=batch_size):
            break
        if (i < 190) and (len(files) > 1) and (i==inds[j]): 
            flair_img = nib.load(root + '/' + files[-5]).get_data()
            inData[j,:,:,:,0] = flair_img[40:-40,40:-40,4:-7]

            t1_img = nib.load(root + '/' + files[-3]).get_data()
            inData[j,:,:,:,1] = t1_img[40:-40,40:-40,4:-7]

            t1ce_img = nib.load(root + '/' + files[-2]).get_data()
            inData[j,:,:,:,2] = t1ce_img[40:-40,40:-40,4:-7]

            t2_img = nib.load(root + '/' + files[-1]).get_data()
            inData[j,:,:,:,3] = t2_img[40:-40,40:-40,4:-7]

            seg = nib.load(root + '/' + files[-4]).get_data()
            seg_img = seg
            
            yFull[j,:,:,:] = seg_img
            
            for i in range(5):
                labelData[j,:,:,:,i] = np.equal(seg_img[40:-40,40:-40,4:-7],i)
            
            j+= 1
        i +=1 
        
    for root, dirs, files in os.walk('../LGG/'):
        if (j>=batch_size):
            break
        if (i - 190 <65) and (len(files) > 1) and (i==inds[j]): 
            flair_img = nib.load(root + '/' + files[-5]).get_data()
            inData[j,:,:,:,0] = flair_img[40:-40,40:-40,4:-7]

            t1_img = nib.load(root + '/' + files[-3]).get_data()
            inData[j,:,:,:,1] = t1_img[40:-40,40:-40,4:-7]

            t1ce_img = nib.load(root + '/' + files[-2]).get_data()
            inData[j,:,:,:,2] = t1ce_img[40:-40,40:-40,4:-7]

            t2_img = nib.load(root + '/' + files[-1]).get_data()
            inData[j,:,:,:,3] = t2_img[40:-40,40:-40,4:-7]

            seg = nib.load(root + '/' + files[-4]).get_data()
            seg_img = seg
            
            yFull[j,:,:,:] = seg_img
            for i in range(5):
                labelData[j,:,:,:,i] = np.equal(seg_img[40:-40,40:-40,4:-7],i)
            j+= 1
        i +=1
    
    x = inData
    y = labelData.reshape([-1,160*160*144*5])
    
    return x, y, yFull


# In[75]:

def valDatafeed():
    """
    Generate validation batches for the BRATS data
    
    The image resizing is done as follows (same as in training):
        - 40 slices removed from beginning and end of dims 0 and 1
        - 4 removed from start and 7 removed from end of dim 2
        - Image futher downsampled via mean pooling at beginning of CNN
    
    """
    i = 0
    j = 0
    
    inData = np.zeros((30,160,160,144,4))
    labelData = np.zeros((30,160,160,144,5))
    yFull = np.zeros((30,240,240,155))
    
    for root, dirs, files in os.walk('../HGG/'):
        if (j>=30):
            break
        if (i >= 190) and (len(files) > 1): 
            flair_img = nib.load(root + '/' + files[-5]).get_data()
            inData[j,:,:,:,0] = flair_img[40:-40,40:-40,4:-7]

            t1_img = nib.load(root + '/' + files[-3]).get_data()
            inData[j,:,:,:,1] = t1_img[40:-40,40:-40,4:-7]

            t1ce_img = nib.load(root + '/' + files[-2]).get_data()
            inData[j,:,:,:,2] = t1ce_img[40:-40,40:-40,4:-7]

            t2_img = nib.load(root + '/' + files[-1]).get_data()
            inData[j,:,:,:,3] = t2_img[40:-40,40:-40,4:-7]

            seg = nib.load(root + '/' + files[-4]).get_data()
            seg_img = seg
            
            yFull[j,:,:,:] = seg_img
            
            for i in range(5):
                labelData[j,:,:,:,i] = np.equal(seg_img[40:-40,40:-40,4:-7],i)
            
            j+= 1
        i +=1 
        
    for root, dirs, files in os.walk('../LGG/'):
        if (j>=30):
            break
        if (i - 190 >= 65) and (len(files) > 1): 
            flair_img = nib.load(root + '/' + files[-5]).get_data()
            inData[j,:,:,:,0] = flair_img[40:-40,40:-40,4:-7]

            t1_img = nib.load(root + '/' + files[-3]).get_data()
            inData[j,:,:,:,1] = t1_img[40:-40,40:-40,4:-7]

            t1ce_img = nib.load(root + '/' + files[-2]).get_data()
            inData[j,:,:,:,2] = t1ce_img[40:-40,40:-40,4:-7]

            t2_img = nib.load(root + '/' + files[-1]).get_data()
            inData[j,:,:,:,3] = t2_img[40:-40,40:-40,4:-7]

            seg = nib.load(root + '/' + files[-4]).get_data()
            seg_img = seg
            
            yFull[j,:,:,:] = seg_img
            for i in range(5):
                labelData[j,:,:,:,i] = np.equal(seg_img[40:-40,40:-40,4:-7],i)
            j+= 1
        i +=1
    
    x = inData
    y = labelData.reshape([-1,160*160*144*5])
    
    return x, y, yFull


# In[61]:

def diceCalc(tensor1, tensor2):
    #
    # Subroutine for calculating dice score of two equal-sized binary tensors
    #
    
    shape = tf.shape(tensor1)
    reshape1 = tf.reshape(tensor1,[shape[0],-1])
    reshape2 = tf.reshape(tensor2,[shape[0],-1])
    
    score = 2*tf.reduce_mean(tf.divide(tf.reduce_sum(tf.multiply(reshape1,reshape2),axis=1),
                       (tf.reduce_sum(reshape1,axis=1) + tf.reduce_sum(reshape2,axis=1))))
    
    return score


# In[62]:

def DICEscores(yFull, pred):
    #
    # truth, estimate = tf tensors of size N x Volume size
    # 
    # Computes Dice scores for each class and overall tumor region
    # 
    
    yFullTensor = tf.stack(yFull)
    predFull = tf.pad(pred,[[0,0], [40, 40], [40, 40], [4, 7]])
    
    classScores = []
    
    for i in range(5):
        score = diceCalc(tf.to_float(tf.equal(yFull,i)), tf.to_float(tf.equal(predFull,i)))
        classScores.append(score)
    
    diceScore = diceCalc(tf.to_float(tf.greater(yFull,0)), tf.to_float(tf.greater(predFull,0)))
        
    return diceScore, classScores[0], classScores[1], classScores[2], classScores[3], classScores[4]


# In[82]:

tf.reset_default_graph()

batch_size = 2
l2Param = 1e-5

inputPlaceholder = tf.placeholder(tf.float32, shape = [batch_size, 160,160,144,4])
outputPlaceholder = tf.placeholder(tf.float32, shape = [batch_size, 160*160*144*5])
    
    
x_downsample = tf.layers.average_pooling3d(inputs = inputPlaceholder, pool_size = (2,2,2),
                                strides = (2,2,2), padding='valid',name=None)
conv1 = tf.layers.conv3d(inputs=x_downsample, filters=8, 
                         kernel_size=[5, 5, 5],padding="same", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling3d(inputs = conv1, pool_size = (2,2,2),
                                strides = (2,2,2), padding='valid',name=None)

conv2 = tf.layers.conv3d(inputs=pool1, filters=8, 
                         kernel_size=[5, 5, 5],padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling3d(inputs = conv2, pool_size = (2,2,2),
                                strides = (2,2,2), padding='valid',name=None)

conv3 = tf.layers.conv3d(inputs=pool2, filters=32, 
                         kernel_size=[3, 3, 3],padding="same", activation=tf.nn.relu)
pool3 = tf.layers.max_pooling3d(inputs = conv3, pool_size = (2,2,2),
                                strides = (2,2,2), padding='valid',name=None)

conv4 = tf.layers.conv3d(inputs=pool3, filters=128, 
                         kernel_size=[3, 3, 3],padding="same", activation=tf.nn.relu)

            
W4 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 128], stddev=0.1))
deconv4 = tf.nn.conv3d_transpose(conv4, filter = W4, output_shape = [batch_size,20, 20, 18, 16], 
                                 strides = [1,2,2,2,1])

b4 = tf.Variable(tf.constant(0.1, shape=[16]))
relu4 = tf.nn.relu(deconv4 + b4)

W3 = tf.Variable(tf.truncated_normal([3, 3, 3, 8, 16], stddev=0.1))
deconv3 = tf.nn.conv3d_transpose(relu4, filter = W3, output_shape = [batch_size,40, 40, 36, 8], 
                                 strides = [1,2,2,2,1])
b3 = tf.Variable(tf.constant(0.1, shape=[8]))
relu3 = tf.nn.relu(deconv3 + b3)

W2 = tf.Variable(tf.truncated_normal([3, 3, 3, 8, 8], stddev=0.1))
deconv2 = tf.nn.conv3d_transpose(relu3, filter = W2, output_shape = [batch_size,80, 80, 72, 8], 
                                 strides = [1,2,2,2,1])
b2 = tf.Variable(tf.constant(0.1, shape=[8]))
relu2 = tf.nn.relu(deconv2 + b2)

W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 5, 8], stddev=0.1))
deconv1 = tf.nn.conv3d_transpose(relu2, filter = W1, output_shape = [batch_size,160, 160, 144, 5], 
                                 strides = [1,2,2,2,1])
b1 = tf.Variable(tf.constant(0.1, shape=[5]))
scores = tf.reshape(deconv1 + b1,[batch_size,5*160*160*144])

    


# In[83]:

# Define loss and training step    
segLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(outputPlaceholder, scores))
regLoss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)     
loss = segLoss + l2Param*regLoss

train_step = tf.train.AdamOptimizer(learning_rate = 1e-3, beta1 = 0.5).minimize(loss)


# Set up functions to evaluate accuracy
probs = tf.nn.softmax(tf.reshape(scores, [batch_size,160,160,144,5]),dim = 4)
prediction = tf.argmax(probs, 4)


# In[ ]:

# Activate session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# Calculate training loop parameters
epoch_num = 1
iterNum = np.int(np.ceil(255/batch_size * epoch_num))


# Run training loop
lossList = []
diceListTrain = []
diceClassListTrain = []
diceListVal = []
diceClassListVal = []

for i in range(iterNum):
    x, y, yFull = trainDatafeed(batch_size)
    pred, loss, _ = sess.run([prediction, segLoss, train_step], feed_dict={inputPlaceholder: x, outputPlaceholder: y})
    
    lossList.append(loss.eval())
    
    groundTruth = tf.convert_to_tensor(np.argmax(y,axis=4))
    dice, scoresList = DICEscore(tf.stack(yFull), pred)
    score0, score1, score2, score3, score4 = scoresList
    
    diceListTrain.append(dice)
    diceClassListTrain.append([score0, score1, score2, score3, score4])
    
    if i % 50 == 0:
        print("Training dice score is: %f" % dice)
        
        x, y, yFull = valDatafeed()
        pred = sess.run(prediction, feed_dict={inputPlaceholder: x, outputPlaceholder: y})
        dice,scoresList = DICEscore(tf.stack(yFull), pred)
        score0, score1, score2, score3, score4 = scoresList
    
        diceListVal.append(dice)
        diceClassListVal.append([score0, score1, score2, score3, score4])
        
        


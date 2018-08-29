import numpy as np
import pandas as pd
import os
import keras
from utils.generator import Generator
from utils.custom_fit_generator import custom_fit_generator
from TCGA_Datasets import TCGA_Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve
from keras import regularizers
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv1D, Flatten,Conv2D, GlobalMaxPooling2D

from keras.optimizers import Adam, RMSprop, Nadam
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.applications.densenet import DenseNet169
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Model(object):
    
    def __init__(self, config):
        
        self.config = config
        self.data_init()
        self.model_init()
        
    def data_init(self):
        
        print("\nData init")
        self.dataset = TCGA_Dataset(self.config)
        generator = Generator(self.config, self.dataset)
        self.train_generator = generator.generate()
        
        self.X_val, self.y_val = self.dataset.convert_to_arrays(self.dataset._partition[0]['val'], self.dataset._partition[1]['val'], phase = 'val',  size = self.config.sampling_size_val)
        self.X_test, self.y_test = self.dataset.convert_to_arrays(self.dataset._partition[0]['test'], self.dataset._partition[1]['test'], phase = 'test', size = self.config.sampling_size_test)
        self.y_test = self.patch_to_image(self.y_test, proba=False)   

    def plot_ROCs(self, y_scores):
        
        fig = plt.figure(figsize=(10,10))
        y_true = self.y_test
        y_score = y_scores
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, lw=2, c='r', alpha=0.8, label = r'%s (AUC = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Luck', alpha=.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        fig.savefig("output/ROC_curve")
        plt.close()

    def plot_PRs(self, y_scores):
        
        fig = plt.figure(figsize=(10,10))

        y_true = self.y_test
        y_score = y_scores
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, lw=2, c='b', alpha=0.8, label=r'PR curve (AP = %0.2f)' % (precision))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("PR curve")
        plt.legend(loc="lower right")
        fig.savefig("output/PR_curve")
        plt.close()
       

    def model_init(self):
        
 
        print("\nModel init")
        self.base_model =  DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224,3), pooling= None)
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048,  activation='relu', kernel_regularizer= l2(0.10))(x)
        x = Dropout(0.30)(x)
        x = Dense(100, activation='relu', kernel_regularizer= l2(0.10))(x)
        x = Dropout(0.30)(x)
        #output = Dropout(0.50)(x, training=True)
        output = Dense(1,  activation='sigmoid')(x)
        self.model = keras.models.Model(inputs=self.base_model.input, outputs=output)
        
        
    def set_trainable(self, from_idx=0):
        
        print("\nTraining")
        #for layer in self.base_model.layers:
        #   layer.trainable = False
        for layer in self.model.layers[0:]:
            layer.trainable = True

    def train(self, lr=1e-4, epochs=10, from_idx=0):
        
        self.set_trainable()
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=self.config.lr_decay)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
        train_steps = len(self.dataset._partition[0]['train'])/self.config.batch_size
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.history = custom_fit_generator(model=self.model, generator=self.train_generator, steps_per_epoch=train_steps, epochs=epochs, verbose=1, validation_data=(self.X_val, self.y_val), shuffle=True, max_queue_size=30, workers=30, use_multiprocessing=True)
#callbacks=[early_stopping])
    
    def predict(self):
        
        print("\nPredicting")
        y_scores = self.model.predict(self.X_test, batch_size= self.config.batch_size) 
       # y_preds = np.array([(y_score>0.5).astype(int) for y_score in y_scores]).flatten()
   
        y_scores = self.patch_to_image(y_scores, proba=True)
        print('y_scores', y_scores)
        #y_preds = self.patch_to_image(y_preds, proba=False)
        y_preds = np.array([(y_score>0.5).astype(int) for y_score in y_scores]).flatten()
        #y_preds = self.patch_to_image(y_preds, proba=False)
        pd.DataFrame(data = y_preds, index =self.dataset._partition[0]['test'] ).to_csv('Results.csv')
        print(self.dataset._partition[0]['test'], y_preds)
        return y_scores, y_preds
    
    def train_predict(self):
        
        self.train(self.config.lr, self.config.epochs, self.config.from_idx)
       # self.plot_loss()
        y_scores, y_preds = self.predict()
        np.save("output/y_scores", y_scores)
        np.save("output/y_preds", y_preds)
        
        return y_scores, y_preds
    
    def patch_to_image(self, y_patches, proba=True):
        
        if proba == True:
            y_image = np.array([np.mean(y_patches[i*self.config.sampling_size_test:(i+1)*self.config.sampling_size_test])for i in range(int(len(y_patches)/self.config.sampling_size_test))]).reshape((-1,1))
        else:
            y_image = np.array([np.mean(y_patches[i*self.config.sampling_size_test:(i+1)*self.config.sampling_size_test])>0.5 for i in range(int(len(y_patches)/self.config.sampling_size_test))]).reshape((-1,1)).astype(int)
        y_image = np.asarray(y_image.flatten())
        return y_image
    
    def plot_loss(self):
        
        keys = list(self.history.history.keys())
        val_acc_keys = [key for key in keys if key[0:3]=="val" and key[-3:]=="acc"]
        acc_keys = [key for key in keys if key[0:3]!="val" and key[-3:]=="acc"]
        val_acc = np.mean([self.history.history[key] for key in val_acc_keys], axis=0)
        acc = np.mean([self.history.history[key] for key in acc_keys], axis=0)
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]

        fig = plt.figure(figsize = (10,10))
        ax1 = plt.subplot(121)
        ax1.tick_params(labelsize=10)
        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('Mean accuracy', size=14)
        plt.ylabel('accuracy', size=12)
        plt.xlabel('epoch', size=12)
        plt.legend(['train', 'test'], loc='upper left', fontsize=12) 
        ax2 = plt.subplot(122)
        ax2.tick_params(labelsize=10)
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('Mean loss', size=14)
        plt.ylabel('loss', size=12)
        plt.xlabel('epoch', size=12)
        plt.legend(['train', 'test'], loc='upper left', fontsize=12)
        plt.show()
        fig.savefig("output/learning_curve")
        plt.close()

    def plot(self):
    
        print("\nPlotting model")
        plot_model(self.model, to_file='output/model.png')

        
    def get_metrics(self, y_scores, y_preds):       
        list_of_metrics = ["accuracy", "precision", "recall", "f1score", "AUC", "AP"]
        self.metrics = pd.DataFrame(data=np.zeros((1, len(list_of_metrics))),columns=list_of_metrics)
        y_true = self.y_test
        y_pred = y_preds
        y_score = y_scores
        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        print(accuracy)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1score = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        self.metrics.iloc[:,:] = [accuracy, precision, recall, f1score, auc, avg_precision]
        self.metrics.to_csv("output/metrics.csv")
        
        

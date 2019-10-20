import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

import numpy as np
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Dense, Lambda, Activation, LSTM, Reshape, Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, concatenate, RepeatVector, multiply,Add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adagrad, Adam, SGD, RMSprop, Nadam
from keras.regularizers import l2

from Dataset_hr_4type import Dataset
from evaluate_hr_4type import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
import scipy.sparse as sp
import gc

def slice(x, index):
    return x[:, index, :, :]


def slice_2(x, index):
    return x[:, index, :]

def get_embedding(path_type, input1, path_num, jumps, length):
    
    conv = Conv1D(filters = 16,
                       kernel_size = 4,
                       activation = 'relu',
                       kernel_regularizer = l2(0.0),
                       kernel_initializer = 'glorot_uniform',
                       padding = 'valid',
                       strides = 1,
                       name = 'uuul_'+path_type)

    path_input = Lambda(slice, output_shape=(jumps, length), arguments={'index':0})(input1)
    output = GlobalMaxPooling1D()(conv(path_input))
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(jumps, length), arguments={'index':i})(input1)
        tmp_output = GlobalMaxPooling1D()(conv(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])
    
    output = Reshape((path_num, 16))(output)
    output = GlobalMaxPooling1D()(output)
    return output

def attention1(latent_name ,latent, ulul_latent, ulll_latent, uuul_latent, uull_latent): 
    output1 = []
    path_output = concatenate([ulul_latent, ulll_latent, uuul_latent, uull_latent])
    path_output = Reshape((4, 16))(path_output)
    path_num = path_output.shape[1].value
    latent_size = latent.shape[1].value
    for i in range(0, path_num):
        metapath = Lambda(slice_2, output_shape=(latent_size,), arguments={'index':i})(path_output)
        tmp_output =  multiply([metapath, latent])
        tmp_output =  Lambda(lambda x: K.sum(x,1,keepdims=True))(tmp_output)
        tmp_output = concatenate([tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output,tmp_output])
        atten = Lambda(lambda x : K.softmax(x), name = latent_name + '_attention_softmax%d'%i)(tmp_output)
        output1.append(atten)
    new_tensor = concatenate(output1)
    print('new_tensor1:',new_tensor)
    new_tensor = Reshape((path_num, 16))(new_tensor)
    print('new_tensor1:',new_tensor)
   
    output = Lambda(lambda x: K.sum(multiply([x[0], x[1]]), 1))([path_output, new_tensor])
    return output

def attention2(latent_name, latent, path_output):
    latent_size = latent.shape[1].value
    inputs = concatenate([latent, path_output])
    output = Dense(latent_size,
                   activation = 'relu',
                   kernel_initializer = 'glorot_normal',
                   kernel_regularizer =l2(0.001),
                   name = latent_name + '_attention_layer')(inputs)
    atten = Lambda(lambda x : K.softmax(x), name = latent_name + '_attention_softmax')(output)
    output = multiply([latent, atten])
    
    return output

def get_model(usize, isize, path_nums, timestamps, length, layers = [20, 10], reg_layers = [0, 0], latent_dim = 40, reg_latent = 0):
    user_input = Input(shape = (1,), dtype = 'int32', name = 'user_input', sparse = False)
    item_input = Input(shape = (1,), dtype = 'int32', name = 'item_input', sparse = False)
    ulul_input = Input(shape = (path_nums[0], timestamps[0], length,), dtype = 'float32', name = 'ulul_input') 
    ulll_input = Input(shape = (path_nums[1], timestamps[1], length,), dtype = 'float32', name = 'ulll_input')
    uuul_input = Input(shape = (path_nums[2], timestamps[2], length, ), dtype = 'float32', name = 'uuul_input')
    uull_input = Input(shape = (path_nums[3], timestamps[3], length, ), dtype = 'float32', name = 'uull_input')
    User_Feedback = Embedding(input_dim = usize, output_dim = latent_dim, input_length = 1, embeddings_initializer = 'glorot_normal', name = 'user_feedback_embedding')
    
    Item_Feedback = Embedding(input_dim = isize, output_dim = latent_dim, input_length = 1, embeddings_initializer = 'glorot_normal', name = 'item_feedback_embedding')
    user_latent = Reshape((latent_dim,))(Flatten()(User_Feedback(user_input)))
    item_latent = Reshape((latent_dim,))(Flatten()(Item_Feedback(item_input)))
    
    ulul_latent = get_embedding('ulul', ulul_input, path_nums[0], timestamps[0], length)
    ulll_latent = get_embedding('ulll', ulll_input, path_nums[1], timestamps[1], length)
    uuul_latent = get_embedding('uuul', uuul_input, path_nums[2], timestamps[2], length)
    uull_latent = get_embedding('uull', uull_input, path_nums[3], timestamps[3], length)
    
    path_output1 = attention1('user', user_latent, ulul_latent, ulll_latent, uuul_latent,uull_latent)
    path_output2 = attention1('item', item_latent, ulul_latent, ulll_latent, uuul_latent,uull_latent)
    path_output = Add()([path_output1,path_output2])    
    user_atten = attention2('user',user_latent, path_output)
    item_atten = attention2('item',item_latent, path_output)
    output = concatenate([user_atten, path_output, item_atten])
    
    for idx in range(0, len(layers)):
        layer = Dense(layers[idx],
                      kernel_regularizer = l2(0.001),
                      kernel_initializer = 'glorot_normal',
                      activation = 'relu',
                      name = 'item_layer%d' % idx)
        output = layer(output)

    print ('output.shape = ', output.shape)
    prediction_layer = Dense(1, 
                       activation = 'sigmoid',
                       kernel_initializer = 'lecun_normal',
                       name = 'prediction')

    prediction = prediction_layer(output)
    model = Model(inputs = [user_input, item_input, ulul_input, ulll_input, uuul_input, uull_input], outputs = [prediction])

    return model

def get_train_instances(user_feature, item_feature, path_ulul, path_ulll, path_uuul,  path_uull, path_nums, jumps, train_list, num_negatives, batch_size, shuffle = True):
   
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1  
    print("train_list",len(train_list))
    print("batch_size",batch_size)
    print("num_batches:",num_batches_per_epoch)
    def data_generator():
        data_size = len(train_list)
        while True:
            if shuffle == True:
                np.random.shuffle(train_list)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                k = 0
                _user_input = np.zeros((batch_size * (num_negatives + 1),))
                _item_input = np.zeros((batch_size * (num_negatives + 1),))
                _ulul_input = np.zeros((batch_size * (num_negatives + 1), path_nums[0], jumps[0], 8))
                _ulll_input = np.zeros((batch_size * (num_negatives + 1), path_nums[1], jumps[1], 8))
                _uuul_input = np.zeros((batch_size * (num_negatives + 1), path_nums[2], jumps[2], 8))
                _uull_input = np.zeros((batch_size * (num_negatives + 1), path_nums[3], jumps[3], 8))
                _labels = np.zeros(batch_size * (num_negatives + 1))
                for u, i in train_list[start_index : end_index]:

                    _user_input[k] = u
                    _item_input[k] = i
                    if (u, i) in path_ulul:

                        for p_i in range(len(path_ulul[(u, i)])):
                            for p_j in range(len(path_ulul[(u, i)][p_i])):
                                type_id = path_ulul[(u, i)][p_i][p_j][0]
                                index = path_ulul[(u, i)][p_i][p_j][1]
                                if type_id == 1 :
                                    _ulul_input[k][p_i][p_j] = user_feature[index] 
                                elif type_id == 2 :
                                    _ulul_input[k][p_i][p_j] = item_feature[index]
                               
                    if (u, i) in path_ulll:

                        for p_i in range(len(path_ulll[(u, i)])):
                            for p_j in range(len(path_ulll[(u, i)][p_i])):
                                type_id = path_ulll[(u, i)][p_i][p_j][0];
                                index = path_ulll[(u, i)][p_i][p_j][1]
                                if type_id == 1 :
                                    _ulll_input[k][p_i][p_j] = user_feature[index] 
                                elif type_id == 2 :
                                    _ulll_input[k][p_i][p_j] = item_feature[index] 
                               
                    if (u, i) in path_uuul:

                        for p_i in range(len(path_uuul[(u, i)])):
                            for p_j in range(len(path_uuul[(u, i)][p_i])):
                                type_id = path_uuul[(u, i)][p_i][p_j][0];
                                index = path_uuul[(u, i)][p_i][p_j][1]
                                if type_id == 1 :
                                    _uuul_input[k][p_i][p_j] = user_feature[index] 
                                elif type_id == 2 :
                                    _uuul_input[k][p_i][p_j] = item_feature[index] 
                                
                    if (u, i) in path_uull:

                        for p_i in range(len(path_uull[(u, i)])):
                            for p_j in range(len(path_uull[(u, i)][p_i])):
                                type_id = path_uull[(u, i)][p_i][p_j][0];
                                index = path_uull[(u, i)][p_i][p_j][1]
                                if type_id == 1 :
                                    _uull_input[k][p_i][p_j] = user_feature[index] 
                                elif type_id == 2 :
                                    _uull_input[k][p_i][p_j] = item_feature[index] 
                            
                    _labels[k] = 1.0
                    k += 1

                    for t in range(num_negatives):
                        j = np.random.randint(1, num_items)
                        while j in user_item_map[u]:
                            j = np.random.randint(1, num_items)
                        
                        _user_input[k] = u
                        _item_input[k] = j
            
                            
                        if (u, j) in path_ulul:

                            for p_i in range(len(path_ulul[(u, j)])):
                                for p_j in range(len(path_ulul[(u, j)][p_i])):
                                    type_id = path_ulul[(u, j)][p_i][p_j][0]
                                    index = path_ulul[(u, j)][p_i][p_j][1]
                                    if type_id == 1 :
                                        _ulul_input[k][p_i][p_j] = user_feature[index] 
                                    elif type_id == 2 :
                                        _ulul_input[k][p_i][p_j] = item_feature[index]
                               
                        if (u, j) in path_ulll:

                            for p_i in range(len(path_ulll[(u, j)])):
                                for p_j in range(len(path_ulll[(u, j)][p_i])):
                                    type_id = path_ulll[(u, j)][p_i][p_j][0];
                                    index = path_ulll[(u, j)][p_i][p_j][1]
                                    if type_id == 1 :
                                        _ulll_input[k][p_i][p_j] = user_feature[index] 
                                    elif type_id == 2 :
                                        _ulll_input[k][p_i][p_j] = item_feature[index]
                               
                        if (u, j) in path_uuul:

                            for p_i in range(len(path_uuul[(u, j)])):
                                for p_j in range(len(path_uuul[(u, j)][p_i])):
                                    type_id = path_uuul[(u, j)][p_i][p_j][0];
                                    index = path_uuul[(u, j)][p_i][p_j][1]
                                    if type_id == 1 :
                                        _uuul_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2 :
                                        _uuul_input[k][p_i][p_j] = item_feature[index]
                                        
                        if (u, j) in path_uull:

                            for p_i in range(len(path_uull[(u, j)])):
                                for p_j in range(len(path_uull[(u, j)][p_i])):
                                    type_id = path_uull[(u, j)][p_i][p_j][0];
                                    index = path_uull[(u, j)][p_i][p_j][1]
                                    if type_id == 1 :
                                        _uull_input[k][p_i][p_j] = user_feature[index] 
                                    elif type_id == 2 :
                                        _uull_input[k][p_i][p_j] = item_feature[index] 
                        
                        _labels[k] = 0.0
                        k += 1
                    
                yield ([_user_input, _item_input, _ulul_input, _ulll_input, _uuul_input, _uull_input], _labels)
    return num_batches_per_epoch, data_generator()
    
if __name__ == '__main__':

    latent_dim = 16
    reg_latent = 0
    layers = [128, 64,32,16]
    reg_layes = [0 ,0, 0, 0]
    learning_rate = 0.001
    epochs = 41
    batch_size = 256
    num_negatives = 4
    learner = 'adam'
    verbose = 1
    out = 0
    evaluation_threads = 1
    topK = 5
    
    print ('num_negatives = ', num_negatives)

    t1 = time()
    dataset = Dataset('tmp/')
    
    trainMatrix, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    train = dataset.train
    user_item_map = dataset.user_item_map
    item_user_map = dataset.item_user_map
    path_ulul = dataset.path_ulul
    path_ulll = dataset.path_ulll
    path_uuul = dataset.path_uuul
    path_uull = dataset.path_uull
    
    user_feature, item_feature = dataset.user_feature, dataset.item_feature
    print('user_feature',user_feature[1])
    num_users, num_items = trainMatrix.shape[0], trainMatrix.shape[1]
    path_nums = [dataset.ulul_path_num, dataset.ulll_path_num, dataset.uuul_path_num, dataset.uull_path_num]
    jumps = [dataset.ulul_jump, dataset.ulll_jump, dataset.uuul_jump, dataset.uull_jump]

    length = dataset.fea_size
    print('full path')
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (time()-t1, num_users, num_items, len(train), len(testRatings)))
    print ('path nums = ', path_nums)
    model = get_model(num_users, num_items, path_nums, jumps, length, layers, reg_layes, latent_dim, reg_latent)
    model.compile(optimizer = Adam(lr = learning_rate, decay = 1e-6),
                  loss = 'binary_crossentropy')  
    p_list, ndcg_list, p_list1, ndcg_list1, p_list2, ndcg_list2 = [], [], [] , [], [], []
    print ('Begin training....')
    
    for epoch in range(epochs):
        t1 = time()
        train_steps, train_batches = get_train_instances(user_feature, item_feature, path_ulul, path_ulll, path_uuul, path_uull, path_nums, jumps, dataset.train, num_negatives, batch_size, True)
        t = time()
        print ('[%.1f s] epoch %d train_steps %d' % (t - t1, epoch, train_steps))
        #Training
        hist = model.fit_generator(train_batches,
                                   train_steps,
                                   epochs = 1,
                                   verbose = 0)
        print ('training time %.1f s' % (time() - t))
        t2 = time()
        if epoch > 5:
            
           (p, ndcg, p1, ndcg1, p2, ndcg2) = evaluate_model(model, user_feature, item_feature, num_users, num_items, path_ulul, path_ulll, path_uuul, path_uull, path_nums, jumps, length, testRatings, testNegatives, topK, evaluation_threads)
           p, ndcg, p1, ndcg1, p2, ndcg2, loss = np.array(p).mean(), np.array(ndcg).mean(), np.array(p1).mean(), np.array(ndcg1).mean(), np.array(p2).mean(), np.array(ndcg2).mean(), hist.history['loss'][0]
           print('Iteration %d [%.1f s]: HR3 = %.4f, Ndcg3 = %.4f, HR5 = %.4f, Ndcg5 = %.4f, HR10 = %.4f, Ndcg10 = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, p, ndcg, p1, ndcg1, p2, ndcg2, loss, time()-t2))
		 
           p_list.append(p)
           ndcg_list.append(ndcg)
           p_list1.append(p1)
           ndcg_list1.append(ndcg1)
           p_list2.append(p2)
           ndcg_list2.append(ndcg2)

    print("End. hr3 = %.4f, ndcg3 = %.4f,  hr5 = %.4f, ndcg5 = %.4f, hr10 = %.4f, ndcg10 = %.4f. " %(max(np.array(p_list)), max(np.array(ndcg_list)),max(np.array(p_list1)), max(np.array(ndcg_list1)),max(np.array(p_list2)), max(np.array(ndcg_list2))))

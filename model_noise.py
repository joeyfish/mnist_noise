from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization , Flatten
from keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


import platform as plat
import os, sys
import time
import matplotlib.pyplot as plt  

import keras as kr
import numpy as np
import random

from readdata import DataMnist

abspath = ''
ModelName='002'
predict_size = 10
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_LEN = IMG_WIDTH * IMG_HEIGHT

batch_size = 128
nb_classes = 10
nb_epoch = 2
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

defsave_step = 500

#reference URL:https://blog.csdn.net/briblue/article/details/80398369 
#ref 基于Keras+CNN的MNIST数据集手写数字分类 - 简书  https://www.jianshu.com/p/3a8b310227e6
class ModelMnist():
	def __init__(self, datapath):
		MS_OUTPUT_SIZE = 10
		self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE # 神经网络最终输出的每一个字符向量维度的大小
		self.label_max_string_length = 64
		self._model =  self.CreateModel() 
		self.base_model = self._model
		self.datapath = datapath

	'''
	def CreateModel(self):

		self._model, self.CreateModel() 
		self.base_model = self._model
	'''	
		
	def CreateModel(self):
		model = Sequential()
		model.add(Convolution2D(nb_filters,
														kernel_size[0],
														kernel_size[1],
														border_mode='valid',
														input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=pool_size))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy',
									optimizer='adadelta',
									metrics=['accuracy'])
		model.summary()              
		return model
	
	def restoreFromLastPoint(self, ModelName, save_step):
		if(not os.path.exists('step'+ModelName+'.txt')):  
			print("return 0")
			return 0
		f = open('step'+ModelName+'.txt','r')
		txt_mode_name = f.read()
		f.close()
		self.LoadModel(txt_mode_name + '.model')
		print(txt_mode_name)
		#for example, get mnist_model/m002/speech_model002_e_0_step_28000
		#need return 28000 / save_step = 28000 / 500
		txt_lines=txt_mode_name.split('_')
		print('****restoreFromLastPoint**** ' , txt_lines[-1], int(txt_lines[-1])//save_step)
		return int(txt_lines[-1])//save_step
			
	def TrainModel(self, datapath, epoch = 8, save_step = 1000, batch_size = 100, filename = abspath + 'mnist_model/m' + ModelName + '/speech_model'+ModelName):
		'''
		训练模型
		参数：
			datapath: 数据保存的路径
			epoch: 迭代轮数
			save_step: 每多少步保存一次模型
			filename: 默认保存文件名，不含文件后缀名
		'''
		n_step = self.restoreFromLastPoint(ModelName, save_step)
		data=DataMnist(datapath, type = 'train', addnoise = True)
		
		num_data = data.GetDataNum() # 获取数据的数量
		
		yielddatas = data.data_genetator_4_mnist_noise(batch_size, IMG_LEN)
		
		for epoch in range(epoch): # 迭代轮数
			print('[running] train epoch %d .' % epoch)
			#n_step = 0 # 迭代数据数
			while True:
				try:
					print('[message] epoch %d . Have train datas %d+'%(epoch, n_step*save_step))
					# data_genetator是一个生成器函数
					
					#self._model.fit_generator(yielddatas, save_step, nb_worker=2)
					self._model.fit_generator(yielddatas, save_step)
					n_step += 1
				except StopIteration:
					print('[error] generator error. please check data format.')
					break
				
				self.SaveModel(comment='_e_'+str(epoch)+'_step_'+str(n_step * save_step))
				self.TestModel(self.datapath, str_dataset='train', data_count = 4)
				self.TestModel(self.datapath, str_dataset='dev', data_count = 4)

	def TestModel(self, datapath='', str_dataset='dev', data_count = 32, out_report = False, show_ratio = True, io_step_print = 10, io_step_file = 10):
		'''
		测试检验模型效果
		
		io_step_print
			为了减少测试时标准输出的io开销，可以通过调整这个参数来实现
		
		io_step_file
			为了减少测试时文件读写的io开销，可以通过调整这个参数来实现
		
		'''
		data=DataMnist(self.datapath, type = str_dataset, addnoise = True)
		#data.LoadDataList(str_dataset) 
		num_data = data.GetDataNum() # 获取数据的数量
		if(data_count <= 0 or data_count > num_data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
			data_count = num_data
		
		try:
			ran_num = random.randint(0,num_data - 1) # 获取一个随机数
			
			words_num = 0
			word_error_num = 0
			
			nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
			if(out_report == True):
				txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8') # 打开文件并读入
			
			txt = '测试报告\n模型编号 ' + ModelName + '\n\n'
			for i in range(data_count):
				data_input, data_labels = data.GetData((ran_num + i) % num_data, 2)  # 从随机数开始连续向后取一定数量数据
				
				# 数据格式出错处理 开始
				# 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
				num_bias = 0
				while(data_input.shape[0] > IMG_LEN):
					print('*[Error]','wave data lenghth of num',(ran_num + i) % num_data, 'is too long.','\n A Exception raise when test Speech Model.')
					num_bias += 1
					data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)  # 从随机数开始连续向后取一定数量数据
				# 数据格式出错处理 结束
				
				pre = self.Predict(data_input, data_input.shape[0] // 8)
				#print("pre=", (pre))
				words_n = data_labels.shape[0] # 获取每个句子的字数
				words_num += words_n # 把句子的总字数加上
				edit_distance = GetEditDistance(data_labels, pre) # 获取编辑距离
				if(edit_distance <= words_n): # 当编辑距离小于等于句子字数时
					word_error_num += edit_distance # 使用编辑距离作为错误字数
				else: # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
					word_error_num += words_n # 就直接加句子本来的总字数就好了
				
				if((i % io_step_print == 0 or i == data_count - 1) and show_ratio == True):
					#print('测试进度：',i,'/',data_count)
					print('Test Count: ',i,'/',data_count)
				
				
				if(out_report == True):
					if(i % io_step_file == 0 or i == data_count - 1):
						txt_obj.write(txt)
						txt = ''
					
					txt += str(i) + '\n'
					txt += 'True:\t' + str(data_labels) + '\n'
					txt += 'Pred:\t' + str(pre) + '\n'
					txt += '\n'
					
				
			
			#print('*[测试结果] 识别 ' + str_dataset + ' 集单字错误率：', word_error_num / words_num * 100, '%')
			print('*[Test Result] Speech Recognition ' + str_dataset + ' set word error ratio: ', word_error_num / words_num * 100, '%')
			if(out_report == True):
				txt += '*[测试结果] 识别 ' + str_dataset + ' 集单字错误率： ' + str(word_error_num / words_num * 100) + ' %'
				txt_obj.write(txt)
				txt = ''
				txt_obj.close()
			
		except StopIteration:
			print('[Error] Model Test Error. please check data format.')

	def Predict(self, data_input, input_len):
		'''
		预测结果
		返回语音识别后的拼音符号列表
		'''
		
		batch_size = 1 
		in_len = np.zeros((batch_size),dtype = np.int32)
		
		in_len[0] = input_len
		
		x_in = np.zeros((batch_size, 28, 28, 1), dtype=np.float)
		
		for i in range(batch_size):
			x_in[i,0:len(data_input)] = data_input
		
		
		base_pred = self.base_model.predict(x = x_in)
		
		#print('base_pred:\n', base_pred)
		
		#y_p = base_pred
		#for j in range(200):
		#	mean = np.sum(y_p[0][j]) / y_p[0][j].shape[0]
		#	print('max y_p:',np.max(y_p[0][j]),'min y_p:',np.min(y_p[0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][j][100])
		#	print('argmin:',np.argmin(y_p[0][j]),'argmax:',np.argmax(y_p[0][j]))
		#	count=0
		#	for i in range(y_p[0][j].shape[0]):
		#		if(y_p[0][j][i] < mean):
		#			count += 1
		#	print('count:',count)
		
		#base_pred =base_pred[:]
		#print('base_pred:',base_pred)
		maxv = 0
		imax = 0
		for j in range(base_pred.shape[1]):
			if maxv < base_pred[0][j]:
				maxv = base_pred[0][j]
				imax = j 
		print('imax:',imax)
		#base_pred =base_pred[:, 2:, :]
		
		#r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1)
		
		#print('r', r)
		
		
		#r1 = K.get_value(r[0][0])
		#print('r1', r1)
		
		
		#r2 = K.get_value(r[1])
		#print(r2)
		
		#r1=r1[0]
		r = base_pred[0]
		for j in range(base_pred.shape[1]):
			if j == imax:
				r[j] = 1
			else:
				r[j] = 0	
		return r
		pass

	def LoadModel(self,filename = abspath + 'mnist_model/m'+ModelName+'/speech_model'+ModelName+'.model'):
		'''
		加载模型参数
		'''
		self._model.load_weights(filename)
		self.base_model.load_weights(filename + '.base')

	def SaveModel(self,filename = abspath + 'mnist_model/m'+ModelName+'/speech_model'+ModelName,comment=''):
		'''
		保存模型参数
		'''
		self._model.save_weights(filename + comment + '.model')
		self.base_model.save_weights(filename + comment + '.model.base')
		# 需要安装 hdf5 模块
		self._model.save(filename + comment + '.h5')
		self.base_model.save(filename + comment + '.base.h5')
		f = open('step'+ModelName+'.txt','w')
		f.write(filename+comment)
		f.close()

	  
	@property
	def model(self):
		'''
		返回 model
		'''
		return self._model

def conv2d(size):
		return Conv2D(size, (3,3), use_bias=True, activation='relu',
				padding='same', kernel_initializer='he_normal')


def norm(x):
		return BatchNormalization(axis=-1)(x)


def maxpool(x):
		return MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(x)
	
def dense(units, activation="relu"):
		return Dense(units, activation=activation, use_bias=True,
				kernel_initializer='he_normal')


# x.shape=(none, none, none)
# output.shape = (1/2, 1/2, 1/2)
def cnn_cell(size, x, pool=True):
    x = norm(conv2d(size)(x))
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)
    return x

import difflib

def GetEditDistance(str1, str2):
        leven_cost = 0
        s = difflib.SequenceMatcher(None, str1, str2)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
                #print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
                if tag == 'replace':
                        leven_cost += max(i2-i1, j2-j1)
                elif tag == 'insert':
                        leven_cost += (j2-j1)
                elif tag == 'delete':
                        leven_cost += (i2-i1)
        if leven_cost != 0:
        	print("leven_cost=", (leven_cost))
        return leven_cost

def load_data(to_caterorical=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if to_caterorical:
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test

if(__name__=='__main__'):
	datapath =  abspath + ''
	modelpath =  abspath + 'mnist_model'
	
	
	if(not os.path.exists(modelpath)):  
		os.makedirs(modelpath) 
	'''
	mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
	#validation_images = train_images[:validation_size]
	#validation_labels = train_labels[:validation_size]
	#train_images = train_images[validation_size:]
	#train_labels = train_labels[validation_size:]
	print(mnist.train.images.shape)
	print(mnist.train.labels.shape)
	image = mnist.train.images[11111,:]
	image = image.reshape(28,28)
	
	plt.figure()
	plt.imshow(image)
	#plt.show()
	'''
	
	system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
	if(system_type == 'Windows'):
		datapath = 'E:\\语音数据集'
		modelpath = modelpath + '\\'
	elif(system_type == 'Linux'):
		datapath =  abspath + 'dataset'
		modelpath = modelpath + '/'+'m'+ModelName+'/'
		if(not os.path.exists(modelpath)):  
			os.makedirs(modelpath) 
	else:
		print('*[Message] Unknown System\n')
		datapath = 'dataset'
		modelpath = modelpath + '/'
	
	
	ms = ModelMnist(datapath)
	##test code##############
	#ms.restoreFromLastPoint(ModelName, defsave_step)
	#exit(1)
	#########################
	#ms.TrainModel(datapath, epoch = 8, batch_size = 100, save_step = 500) final value here
	ms.TrainModel(datapath, epoch = 8, batch_size = 100, save_step = defsave_step)

def mnist_gaussian_noise():
    for sigma in [8, 32, 128]:
        x_train, y_train, x_test, y_test = load_data()
        model = ms #get_model()

        #plt.imshow(np.squeeze(x_train[0]), cmap='gray')
        #plt.show()
        #plot_many(0, x_train, y_train, sigma)

        noise = np.random.normal(0, sigma, size=x_train.shape)
        x_train_gaus = x_train + noise
        x_train_gaus = np.clip(x_train_gaus, 0, 255)

        #plt.imshow(np.squeeze(x_train_gaus[0]), cmap='gray')
        #plt.show()
        #plot_many(0, x_train_gaus, y_train, sigma)

        history = model._model.fit(x_train_gaus, y_train, batch_size=batch_size, nb_epoch=1,
                  verbose=1, validation_data=(x_test, y_test))

        result = {}
        result['sigma'] = sigma
        result['score'] = model.evaluate(x_test, y_test, verbose=0)
        result['pred_probs'] = model.predict(x_test, batch_size=32, verbose=0)
        result['cmat'] = get_confusion_matrix(result['pred_probs'], y_test)
        
        results_gaussian.append(result)
        
        
mnist_gaussian_noise()
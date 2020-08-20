#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import platform as plat
import os, sys

import numpy as np
import matplotlib.pyplot as plt  

import random
#import scipy.io.wavfile as wav
predict_size = 10

SHOW_PLT = 1
PRTINT_NSTART = 2
flagAddNoise = 0

class DataMnist():
	
	
	def __init__(self, path, type, LoadToMem = False, MemWavCount = 10000, addnoise = False):
		'''
		初始化
		参数：
			path：数据存放位置根目录
			type:  'train' and 'dev'
		'''
		
		system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
		
		self.datapath = path; # 数据存放位置根目录
		self.type = type # 数据类型，分为三种：训练集(train)、验证集(dev)、测试集(test)
		
		self.slash = ''
		if(system_type == 'Windows'):
			self.slash='\\' # 反斜杠
		elif(system_type == 'Linux'):
			self.slash='/' # 正斜杠
		else:
			print('*[Message] Unknown System\n')
			self.slash='/' # 正斜杠
		
		if(self.slash != self.datapath[-1]): # 在目录路径末尾增加斜杠
			self.datapath = self.datapath + self.slash
		
		
		self.DataNum = 0 # 记录数据量
		self.LoadDataList(addnoise)
		
		self.LoadToMem = LoadToMem
		
		self.getdata_num = 0
		pass
	
	def LoadDataList(self, addnoise):
		'''
		加载用于计算的数据列表
		参数：
			type：选取的数据集类型
				train 训练集
				dev 开发集
				test 测试集
		'''
		mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
		#validation_images = train_images[:validation_size]
		#validation_labels = train_labels[:validation_size]
		#train_images = train_images[validation_size:]
		#train_labels = train_labels[validation_size:]
		print("train=", mnist.train.images.shape)
		print("train=", mnist.train.labels.shape)
		print("test=", mnist.test.images.shape)
		print("test=", mnist.test.labels.shape)
		#image = mnist.train.images[11111,:]
		#image = image.reshape(28,28)
		self.mnist_data = mnist
		self.mnist_data_train_images = mnist.train.images
		self.mnist_data_test_images = mnist.test.images
		#plt.figure()
		#plt.imshow(image)
		#plt.show()
		if addnoise == True:
			self.mnist_data_train_images = self.Add_gaussian_Noise(mnist.train.images)
			self.mnist_data_test_images = self.Add_gaussian_Noise(mnist.test.images)
		self.DataNum = self.GetDataNum()
	
	def GetDataNum(self):
		'''
		获取数据的数量
		当wav数量和symbol数量一致的时候返回正确的值，否则返回-1，代表出错。
		'''
		if self.type == 'train':
			DataNum = self.mnist_data_train_images.shape[0]
		else:	
			DataNum = self.mnist_data_test_images.shape[0]
		return DataNum
		
		
	def GetData(self,n_start,show_plt = 0, n_amount=1):
		'''
		读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
		参数：
			n_start：从编号为n_start数据开始选取数据
			n_amount：选取的数据数量，默认为1，即一次一个wav文件
		返回：
			三个包含wav特征矩阵的神经网络输入值，和一个标定的类别矩阵神经网络输出值
		'''
		self.getdata_num  += 1
		if (show_plt == PRTINT_NSTART):
			print('get data ', n_start)
		if self.type == 'train':	
			image = self.mnist_data_train_images[n_start,:]
		else:		
			image = self.mnist_data_test_images[n_start,:]
		if flagAddNoise == 1:
			image = self.Add_gaussian_Noise(image)	
		img = image.reshape(28,28)
		if show_plt == SHOW_PLT:
			plt.figure()
			plt.imshow(img)
			plt.show()
		image = image.reshape(28,28, 1)
		if self.type == 'train':	
			data_label = self.mnist_data.train.labels[n_start]
		else:	
			data_label = self.mnist_data.test.labels[n_start]
		#aax=np.array([9,8,7,6,5,4,3,2,1,0])
		aax=np.array([0,1,2,3,4,5,6,7,8,9])
		#print('aax=', aax)
		value = np.sum(data_label*aax)
		y_2kinds = np.zeros(2, dtype = np.float)
		y_2kinds[int(value//5)] = 1.
		y_odds = np.zeros(2, dtype = np.float)
		y_odds[int(value%2)] = 1.   #y_odds[0] = 1 if number is even.
		if show_plt == 1:
			print(data_label)
		debugflaghere = False #True
		if debugflaghere:
			print('===============')
			print('get=', self.getdata_num)
			print('value=', value)
			print('data_label=', data_label)
			print('y_2kinds=', y_2kinds)
			print('y_odd=', y_odd)
		return image, data_label, y_2kinds, y_odds
	
	def data_genetator(self, batch_size=100, img_length = 28*28):
		'''
		数据生成器函数，用于Keras的generator_fit训练
		batch_size: 一次产生的数据量
		需要再修改。。。
		'''
		
		#labels = []
		#for i in range(0,batch_size):
		#	#input_length.append([1500])
		#	labels.append([0.0])
		
		
		
		#labels = np.array(labels, dtype = np.float)
		labels = np.zeros((batch_size,1), dtype = np.float)
		aax=np.array([0,1,2,3,4,5,6,7,8,9])
		#print(input_length,len(input_length))
		
		#train_X, train_y = mnist.load_data()[0]
		#train_X = train_X.reshape(-1, 28, 28, 1)
		#train_X = train_X.astype('float32')
		#train_X /= 255
		#train_y = to_categorical(train_y, 10)
		
		
		
		while True:
			X = np.zeros((batch_size, 28, 28, 1), dtype = np.float32)
			#y = np.zeros((batch_size, 64, self.SymbolNum), dtype=np.int16)
			y = np.zeros((batch_size, predict_size), dtype=np.int64)
			y_2kinds = np.zeros((batch_size, 2), dtype=np.int64)
			y_odds = np.zeros((batch_size, 2), dtype=np.int64) 
			
			#generator = ImageCaptcha(width=width, height=height)
			input_length = []
			label_length = []
			
			
			
			for i in range(batch_size):
				ran_num = random.randint(0,self.DataNum - 1) # 获取一个随机数
				#if (i == 0):
				#	print("first ran_num= %d" % (ran_num))
				data_input, data_labels, y_2kind, y_odd = self.GetData(ran_num)  # 通过随机数取一个数据
				#print(data_labels)
				#data_input, data_labels = self.GetData((ran_num + i) % self.DataNum)  # 从随机数开始连续向后取一定数量数据
				
				input_length.append(data_input.shape[0])
				#print(data_input, data_labels)
				#print('data_input长度:',len(data_input))
				
				X[i,0:len(data_input)] = data_input
				#print('data_labels长度:',len(data_labels))
				#print(data_labels)
				y[i,0:len(data_labels)] = data_labels
				y_2kinds[i,0:len(y_2kind)] = y_2kind
				y_odds[i,0:len(y_odd)] = y_odd
				#print(data_labels, y2[i])
				#print(i,y[i].shape)
				#y[i] = y[i].T
				#print(i,y[i].shape)
				#label_length.append([len(data_labels)])
			
			#label_length = np.array(label_length)
			#input_length = np.array(input_length)
			y_2kinds = np.array(y_2kinds)
			y_odds = np.array(y_odds)
			yield X, [y,y_2kinds,y_odds]
			#return X, [y,y_2kinds,y_odds]
			#images, y, y_2kinds, y_odds = [], [], [], []
		pass
		
	def data_genetator_only_big5(self, batch_size=100, img_length = 28*28):
		'''
		数据生成器函数，用于Keras的generator_fit训练
		batch_size: 一次产生的数据量
		需要再修改。。。
		'''
		
		#labels = []
		#for i in range(0,batch_size):
		#	#input_length.append([1500])
		#	labels.append([0.0])
		
		
		
		#labels = np.array(labels, dtype = np.float)
		labels = np.zeros((batch_size,1), dtype = np.float)
		aax=np.array([0,1,2,3,4,5,6,7,8,9])
		#print(input_length,len(input_length))
		
		#train_X, train_y = mnist.load_data()[0]
		#train_X = train_X.reshape(-1, 28, 28, 1)
		#train_X = train_X.astype('float32')
		#train_X /= 255
		#train_y = to_categorical(train_y, 10)
		
		
		
		while True:
			X = np.zeros((batch_size, 28, 28, 1), dtype = np.float32)
			#y = np.zeros((batch_size, 64, self.SymbolNum), dtype=np.int16)
			y = np.zeros((batch_size, predict_size), dtype=np.int64)
			y_2kinds = np.zeros((batch_size, 2), dtype=np.int64)
			y_odds = np.zeros((batch_size, 2), dtype=np.int64) 
			
			#generator = ImageCaptcha(width=width, height=height)
			input_length = []
			label_length = []
			
			
			
			for i in range(batch_size):
				ran_num = random.randint(0,self.DataNum - 1) # 获取一个随机数
				#if (i == 0):
				#	print("first ran_num= %d" % (ran_num))
				data_input, data_labels, y_2kind, y_odd = self.GetData(ran_num)  # 通过随机数取一个数据
				#print(data_labels)
				#data_input, data_labels = self.GetData((ran_num + i) % self.DataNum)  # 从随机数开始连续向后取一定数量数据
				
				input_length.append(data_input.shape[0])
				#print(data_input, data_labels)
				#print('data_input长度:',len(data_input))
				
				X[i,0:len(data_input)] = data_input
				#print('data_labels长度:',len(data_labels))
				#print(data_labels)
				y[i,0:len(data_labels)] = data_labels
				y_2kinds[i,0:len(y_2kind)] = y_2kind
				y_odds[i,0:len(y_odd)] = y_odd
				#print(data_labels, y2[i])
				#print(i,y[i].shape)
				#y[i] = y[i].T
				#print(i,y[i].shape)
				#label_length.append([len(data_labels)])
			
			#label_length = np.array(label_length)
			#input_length = np.array(input_length)
			y_2kinds = np.array(y_2kinds)
			y_odds = np.array(y_odds)
			yield X, [y_2kinds]
			#return X, [y,y_2kinds,y_odds]
			#images, y, y_2kinds, y_odds = [], [], [], []
		pass


	def data_genetator_4_mnist_noise(self, batch_size=100, img_length = 28*28):
		'''
		数据生成器函数，用于Keras的generator_fit训练
		batch_size: 一次产生的数据量
		需要再修改。。。
		'''
		
		#labels = np.array(labels, dtype = np.float)
		labels = np.zeros((batch_size,1), dtype = np.float)
		aax=np.array([0,1,2,3,4,5,6,7,8,9])
		#print(input_length,len(input_length))
		

		
		
		
		while True:
			X = np.zeros((batch_size, 28, 28, 1), dtype = np.float32)
			y = np.zeros((batch_size, predict_size), dtype=np.int16)
			y_2kinds = [] #np.zeros((batch_size), dtype=np.int16) #test code, if smaller than 5, label 10, else label 11
			
			input_length = []
			label_length = []
			
			
			
			for i in range(batch_size):
				ran_num = random.randint(0,self.DataNum - 1) # 获取一个随机数
				data_input, data_labels, y_2kind, y_odd= self.GetData(ran_num)  # 通过随机数取一个数据
				
				input_length.append(data_input.shape[0])
				
				X[i,0:len(data_input)] = data_input
				y[i,0:len(data_labels)] = data_labels
				label_length.append([len(data_labels)])
			
			label_length = np.matrix(label_length)
			input_length = np.array([input_length]).T
			y_2kinds = np.array([y_2kinds]).T
			yield [X], y
		pass


	def GetSymbolList(self):
		'''
		加载拼音符号列表，用于标记符号
		返回一个列表list类型变量
		'''

		return list_symbol

	def GetSymbolNum(self):
		'''
		获取拼音符号数量
		'''
		return len(self.list_symbol)
		
	def SymbolToNum(self,symbol):
		'''
		符号转为数字
		'''
		if(symbol != ''):
			return self.list_symbol.index(symbol)
		return self.SymbolNum
	
	def NumToVector(self,num):
		'''
		数字转为对应的向量
		'''
		v_tmp=[]
		for i in range(0,len(self.list_symbol)):
			if(i==num):
				v_tmp.append(1)
			else:
				v_tmp.append(0)
		v=np.array(v_tmp)
		return v
		
	def Add_gaussian_Noise(self, x_image, sigma = 8):
		img_size = 28*28
		#x_image *= 255
		x_noise = np.random.normal(0, sigma, size=x_image.shape) / 15
		x_noisy_image = x_image + x_noise
		x_noisy_image = np.clip(x_noisy_image, 0, 1)
		return x_noisy_image

	def plot_all_pics(self, start, all_noise, label):    
	    # Create figure with 10 sub-plots.
	    fig, axes = plt.subplots(5, 10)
	    fig.subplots_adjust(hspace=0.2, wspace=0.1)

	    # For each sub-plot.
	    for i, ax in enumerate(axes.flat):
	        # Get the adversarial noise for the i'th target-class.
	        noise_x = all_noise[i + start]
	        noise = noise_x.reshape(28,28)
	        
	        # Plot the noise.
	        ax.imshow(noise,
	                  cmap='seismic', interpolation='nearest',
	                  vmin=-1.0, vmax=1.0)

	        # Show the classes as the label on the x-axis.
	        lab1 = label[i + start]
	        lab = self.categorical_to_numeric(lab1)
	        ax.set_xlabel(lab)

	        # Remove ticks from the plot.
	        ax.set_xticks([])
	        ax.set_yticks([])
	    
	    # Ensure the plot is shown correctly with multiple plots
	    # in a single Notebook cell.
	    plt.show()

	#data from https://github.com/joaquimcodina/NoisedMNIST
	#DATASET_URL = ('https://firebasestorage.googleapis.com/v0/b/hackeps-2019.appspot.com/o/noised-MNIST.npz?alt=media&token=4cee641b-9e31-42c4-b9c8-e771d2eecbad')
	def showpics(self, num):
		#fname = 'noised-MNIST.npz'
		#download_file(DATASET_URL, fname)

		#data = np.load(fname)
		#x, y, x_submission = data.values()
		#print("total nouse=", x.shape)
		#print("y=", y[num])
		'''
		image = x[num]
		img = image.reshape(28,28)
		plt.figure()
		plt.imshow(img)
		plt.show()
		'''
		x = self.mnist_data_train_images
		y = self.mnist_data.train.labels
		self.plot_all_pics(num, x, y)
	
	def categorical_to_numeric(self, label):
		leng = len(label)
		for i in range(leng):
			if label[i] != 0:
				return i
		return 0
			
if(__name__=='__main__'):
	argc = len(sys.argv)
	shownum = 0
	if argc == 2:
		shownum = int(sys.argv[1])
	if argc == 3:
		path='./dataset'
		flagAddNoise = 1
		l=DataMnist(path,type = 'train',  addnoise = True)
		shownum = int(sys.argv[1])
		l.showpics(shownum)
		exit(1)
	print("shownum=", shownum)	



	#path='E:\\语音数据集'
	path='./dataset'
	cnt = 0
	l=DataMnist(path,'train')
	#l.LoadDataList('train')
	print("data num=%d" % (l.GetDataNum()))
	ran_num = random.randint(0,l.GetDataNum() - 1)
	flagAddNoise = 0
	if shownum == 10000:
		flagAddNoise = 1
	elif shownum != 0:
		ran_num = shownum
	#ran_num = shownum
	print("data ran_num=%d" % (ran_num))
	if argc == 2:
		l.showpics(ran_num)
	data_input, data_labels, y_2kind, y_odd = l.GetData(ran_num, 1)
	print('data_input.shape=',data_input.shape)
	print('data_labels=\n',data_labels)
	print('y_2kind=\n',y_2kind)
	print('y_odd=\n',y_odd)
	msg=input("hello1")
	aa=l.data_genetator(15)
	msg=input("hello2")
	for i, im in zip(range(100),aa):
		a,b=im
	#print("size of aa=%d" % (len(aa)))
	msg=input("hello3")
	#[y,y_2kinds,y_odds]
	print('len=',len(b))
	print('shape0=',b[0].shape)
	print('shape1=',b[1].shape)
	print('shape2=',b[2].shape)
	print('y    =',b[0])
	print('2kind=',b[1])
	print('oddev=',b[2])
	pass
	

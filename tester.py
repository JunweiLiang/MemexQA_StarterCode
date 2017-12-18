# coding=utf-8
# author: Junwei Liang junweil@cs.cmu.edu
# tester, given the config with model path


import tensorflow as tf
import numpy as np

class Tester():
	def __init__(self,model,config,sess=None):
		self.config = config
		self.model = model

		self.yp = self.model.yp # the output of the model # [N,M,JX]


	def step(self,sess,batch):
		# give one batch of Dataset, use model to get the result,
		assert isinstance(sess,tf.Session)
		batchIdxs,batch_data =  batch
		feed_dict = self.model.get_feed_dict(batch_data,is_train=False)
		yp, = sess.run([self.yp],feed_dict=feed_dict)
		# clip the output
		# yp should be [N,4]
		yp = yp[:batch_data.num_examples]
		return yp

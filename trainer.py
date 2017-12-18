# coding=utf-8
# author: Junwei Liang junweil@cs.cmu.edu
# trainer class, given the model

import tensorflow as tf


class Trainer():
	def __init__(self,model,config):
		self.config = config
		self.model = model # this is an model instance		

		self.global_step = model.global_step # 

		self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
		#self.opt = tf.train.AdamOptimizer(config.init_lr)

		self.loss = model.loss # get the loss funcion

		# for training, we get the gradients first, then apply them
		self.grads = self.opt.compute_gradients(self.loss) # will train all trainable in Graph
		# process gradients?
		self.train_op = self.opt.apply_gradients(self.grads,global_step=self.global_step)


	def step(self,sess,batch): 
		assert isinstance(sess,tf.Session)
		# idxs is a tuple (23,123,33..) index for sample
		batchIdx,batch_data = batch
		feed_dict = self.model.get_feed_dict(batch_data,is_train=True)
		loss, train_op = sess.run([self.loss,self.train_op],feed_dict=feed_dict)
		return loss, train_op


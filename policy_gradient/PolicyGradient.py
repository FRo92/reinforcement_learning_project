from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class model(tf.keras.Model):
	def __init__(self, gamma):
		super().__init__()
		self.d1 = tf.keras.layers.Dense(30, activation='relu')
		self.d2 = tf.keras.layers.Dense(30, activation='relu')
		self.out = tf.keras.layers.Dense(4,\
                                         activation='softmax')  # output probabilities for each action (Lec 6 Slide 7)
		self.gamma = gamma
		self.opt = Adam(0.0003)
	
		
	def call(self, input_data):
		x = tf.convert_to_tensor(input_data)
		x = self.d1(x)
		x = self.d2(x)
		x = self.out(x)
		return x
	
	def act(self, state):
		"""
        input : numpy array of states
        output: predicted probabilities
        tfp turn probabilities into a distribution
        """
		prob = self(np.array([state]))
		dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
		action = dist.sample()  # sampling from distribution
		return int(action.numpy()[0])  # action returned as integer
	@tf.function
	def a_loss(self, prob, action, reward):
		"""
        loss = -(prob of selected action*total_discount_reward)
        """
		dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
		log_prob = dist.log_prob(action)
		loss = -log_prob * reward  # isch gradient ascent wenn ich mich ned tüsche ned descent
		return loss
	
	def train(self, states, rewards, actions):
		"""
        input: list of states, actions, rewards
        output: calculates gradient
        """
		sum_reward = 0
		G_all = []
		rewards.reverse()  # starting from the last element
		for r in rewards:  # calculates the cumulative expected reward for each state
			sum_reward = r + self.gamma * sum_reward
			G_all.append(sum_reward)
		G_all.reverse()
		
		for state, G_t, action in zip(states, G_all, actions):
			with tf.GradientTape() as g:
				p = self(np.array([state]), training=True)
				loss = self.a_loss(p, action, G_t)
			grads = g.gradient(loss, self.trainable_variables)
			self.opt.apply_gradients(zip(grads,\
                                         self.trainable_variables))
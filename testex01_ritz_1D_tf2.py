import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import os
import shutil
import time
from matplotlib import pyplot as plt 

class Varnn():
    """NNets for PDEs"""

    """
    variational principle for solving PDEs

    Parameters:
    -----------
    spatial_range: spatial domain (square domain), [-1,1] or [0,1], list
    num_hidden: hidden units in each layer, int
    batch_size: batch size, int
    num_iters: the number of iterations, int
    lr_rate: step size (learning rate) for stochastic optimization, float
    output_path: save path for tensorflow model, string
    """

    def __init__(self, spatial_range, num_hidden=20, batch_size=200, num_iters=1000, lr_rate=1e-3, output_path='Varnn'):
        self.spatial_range = spatial_range # for example, spatial range = [-1,1]
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.lr_rate = lr_rate
        self.output_path = output_path
        self.loss_history = []
        tf.reset_default_graph()
        
    def fcnet(self, input):
        """fcnet for all pde_type"""
        num_hidden = self.num_hidden
        with tf.variable_scope("fcnet", reuse=tf.AUTO_REUSE):
            x = tf.layers.Dense(num_hidden, activation=tf.nn.tanh)(input)
            x = tf.layers.Dense(num_hidden, activation=tf.nn.tanh)(x)
            x = x + tf.layers.Dense(num_hidden, activation=tf.nn.tanh)(x)
            x = tf.layers.Dense(num_hidden, activation=tf.nn.tanh)(x)
            output = tf.layers.Dense(1)(x)
        return output

    def alfa(self,x):
        """Set bubble-function for zero boundary condition"""
        a=self.spatial_range[0]
        b=self.spatial_range[1]
        return (x-a)*(b-x)
    
    def beta(self,x):
        """Set function for nonzero boundary condition"""
        return 0

    def train(self):
        """train step"""
        
        z = tf.placeholder(tf.float32, shape=[None,1]) 

        # assign boundary condition to neural networks
        u = self.alfa(z) *self.fcnet(z) + self.beta(z)
        
        gradients = tf.gradients(u, z)
        # first term
        slopes = tf.square(gradients)
            
        def rightside(z):
            return (np.pi**2)*tf.sin(np.pi*z)
   
        # define variational loss for pde
        loss = tf.reduce_mean(0.5*slopes-rightside(z)*u)
        
        train_op = tf.train.AdamOptimizer(self.lr_rate).minimize(loss)
   
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for iter in range(self.num_iters):
                batch_data = self.spatial_range[0] + (self.spatial_range[1]-self.spatial_range[0])*np.random.rand(self.batch_size, 1)
                loss_cur, _ = sess.run([loss, train_op], feed_dict={z: batch_data})
                self.loss_history.append(loss_cur)
                if (iter+1) % 100 == 0:
                    print('iteration: {}, loss: {:.4}'. format(iter+1, loss_cur))
            saver.save(sess, os.path.join(self.output_path, "model"), global_step=iter)
   
    def test(self, x, branch):
        checkpoint = tf.train.latest_checkpoint(self.output_path)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint)
            u = self.alfa(x) *self.fcnet(x) + self.beta(x)
            if branch==1:
               u_test = sess.run(u)
            else:
               gradients = tf.gradients(u, [x])[0] 
               ux=tf.reshape(gradients[:,0],[-1,1])
               u_test = sess.run(ux)
        return u_test
    
 

    def numtest(self, num_points, branch=1):
        xs = np.linspace(self.spatial_range[0],self.spatial_range[1], num_points, dtype=np.float32)
        z=tf.reshape(xs,[-1,1])   
        return self.test(z, branch)

# main program
# Range of Spatial domain, here [0,1]
spatial_range = [0,1]

# units of hidden layer
num_hidden = 20
batch_size = 200
num_iters = 1000

# construct model
VarPDE = Varnn(spatial_range, num_hidden, batch_size, num_iters)
# train model 
tic=time.time()
VarPDE.train()
toc=time.time()
elapsed=toc-tic

# record variational loss
varloss_list = VarPDE.loss_history
np.save('varloss_list.npy', varloss_list)

from scipy.io import savemat
u=VarPDE.numtest(41)
savemat('test01-u.mat', {'u': u})
x=np.linspace(spatial_range[0],spatial_range[1], 41)
plt.plot(x,u)
plt.show()
savemat('test01-loss.mat',{'loss':varloss_list})
plt.plot(varloss_list)
plt.show()

u160=VarPDE.numtest(161)
savemat('test01-u160.mat', {'u': u160})

ux=VarPDE.numtest(41,2)
savemat('test01-ux.mat', {'ux': ux})

plt.plot(x,ux,"r-",x,np.pi*np.cos(np.pi*x),"k-")
plt.show()
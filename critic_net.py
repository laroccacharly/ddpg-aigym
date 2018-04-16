
import numpy as np
import pdb
import tensorflow as tf
import math

from actor_net import Net, weight_init, np_to_var
import torch
import torch.nn as nn
from math import sqrt
from torch.autograd import Variable
from tensorboardX import SummaryWriter
writer = SummaryWriter()

torch.manual_seed(0)
tf.set_random_seed(0)
TAU = 0.001
LEARNING_RATE= 0.001
BATCH_SIZE = 64
CRITIC_WEIGHT_DECAY = 1e-2
HIDDEN_SIZE_1 = 400
HIDDEN_SIZE_2 = 300
CRITIC_LR = 1e-3


class CriticNetPy(Net):
    def __init__(self, nb_features, nb_actions, nb_hidden_1, nb_hidden_2, learning_rate, batch_norm=False):
        super(CriticNetPy, self).__init__()

        self.relu = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm1d(nb_features)
        self.linear_1 = nn.Linear(nb_features, nb_hidden_1)
        self.linear_1.weight.data.uniform_(-1/sqrt(nb_features), 1/sqrt(nb_features))
        self.batch_norm_2 = nn.BatchNorm1d(nb_hidden_1)
        self.linear_2 = nn.Linear(nb_hidden_1 + nb_actions, nb_hidden_2)
        self.linear_2.weight.data.uniform_(-1/sqrt(nb_hidden_1), 1/sqrt(nb_hidden_1))
        self.batch_norm_3 = nn.BatchNorm1d(nb_hidden_2)
        self.linear_3 = nn.Linear(nb_hidden_2, 1)
        self.linear_3.weight.data.uniform_(-3e-3, 3e-3)


        if batch_norm:
            self.layers_1= [
                self.batch_norm_1,
                self.linear_1,
                self.relu,
                self.batch_norm_2,

            ]
            self.layers_2 = [
                self.linear_2,
                self.relu,
                self.batch_norm_3,
                self.linear_3
            ]
        else:
            self.layers_1 = [
                self.linear_1,
                self.relu,

            ]
            self.layers_2 = [
                self.linear_2,
                self.relu,
                self.linear_3
            ]

        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=CRITIC_WEIGHT_DECAY)
        self.step_count = 0

    def forward(self, states, actions):
        #x = torch.cat((states, actions), dim=1)
        #x = np.array([np.concatenate((state, action)) for state, action in zip(states, actions)])
        x = states
        for layer in self.layers_1:
            x = layer(x)

        x = torch.cat((x, actions), dim=1)
        for layer in self.layers_2:
            x = layer(x)

        return x

    def make_summary(self):
        if True:
         return
        writer.add_scalar('critic/linear_1/mean', self.linear_1.weight.mean().data[0], self.step_count)
        writer.add_scalar('critic/linear_2/mean', self.linear_2.weight.mean().data[0], self.step_count)
        writer.add_scalar('critic/linear_3/mean', self.linear_3.weight.mean().data[0], self.step_count)
        writer.add_scalar('critic/linear_1/std', self.linear_1.weight.std().data[0], self.step_count)
        writer.add_scalar('critic/linear_2/std', self.linear_2.weight.std().data[0], self.step_count)
        writer.add_scalar('critic/linear_3/std', self.linear_3.weight.std().data[0], self.step_count)
        self.step_count += 1


class CriticNet:
    """ Critic Q value model of the DDPG algorithm """

    def __init__(self, num_states, num_actions):

        self.critic = CriticNetPy(nb_features=num_states, nb_actions=num_actions , nb_hidden_1=HIDDEN_SIZE_1, nb_hidden_2=HIDDEN_SIZE_2,
                         learning_rate=CRITIC_LR)
        #self.critic.apply(weight_init)
        self.target_critic = CriticNetPy(nb_features=num_states, nb_actions=num_actions, nb_hidden_1=HIDDEN_SIZE_1, nb_hidden_2=HIDDEN_SIZE_2, learning_rate=CRITIC_LR)

        self.target_critic.init_params_from_model(self.critic)

    def evaluate_target_critic(self, state, action):
        state, action = np_to_var(state), np_to_var(action)
        return self.target_critic.forward_to_numpy(state, action)

    def train_critic(self, state, action, y_i_batch):
        state, action, target_action_values = np_to_var(state), np_to_var(action), np_to_var(y_i_batch)

        action_values = self.critic(state, action)
        mse_loss_calculator = nn.MSELoss()
        loss_critic = mse_loss_calculator(action_values, target_action_values) # Detach because no grad on target
        loss_critic.backward()
        #pdb.set_trace()
        self.critic.opt.step()
        self.critic.zero_grad()
        self.critic.make_summary()

    def compute_delQ_a(self, state, action):
        state, action = np_to_var(state), np_to_var(action)
        action.requires_grad = True
        action_values = self.critic(state, action)
        loss = action_values.sum()
        loss.backward()
        return action.grad.data.cpu().numpy()

    def update_target_critic(self):
        self.target_critic.update_params_from_model(self.critic, TAU)


class CriticNetOld:
    """ Critic Q value model of the DDPG algorithm """
    def __init__(self,num_states,num_actions):
        
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #critic_q_model parameters:
            self.W1_c, self.B1_c, self.W2_c, self.W2_action_c, self.B2_c, self.W3_c, self.B3_c,\
            self.critic_q_model, self.critic_state_in, self.critic_action_in = self.create_critic_net(num_states, num_actions)
                                   
            #create target_q_model:
            self.t_W1_c, self.t_B1_c, self.t_W2_c, self.t_W2_action_c, self.t_B2_c, self.t_W3_c, self.t_B3_c,\
            self.t_critic_q_model, self.t_critic_state_in, self.t_critic_action_in = self.create_critic_net(num_states, num_actions)
            
            self.q_value_in=tf.placeholder("float",[None,1]) #supervisor
            #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_c)+tf.nn.l2_loss(self.W2_c)+ tf.nn.l2_loss(self.W2_action_c) + tf.nn.l2_loss(self.W3_c)+tf.nn.l2_loss(self.B1_c)+tf.nn.l2_loss(self.B2_c)+tf.nn.l2_loss(self.B3_c) 
            self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.W2_c,2))+ 0.0001*tf.reduce_sum(tf.pow(self.B2_c,2))             
            self.cost=tf.pow(self.critic_q_model-self.q_value_in,2)/BATCH_SIZE + self.l2_regularizer_loss#/tf.to_float(tf.shape(self.q_value_in)[0])
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)
            
            #action gradient to be used in actor network:
            #self.action_gradients=tf.gradients(self.critic_q_model,self.critic_action_in)
            #from simple actor net:
            self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in)
            self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])][0] #this is just divided by batch size
            #from simple actor net:
            self.check_fl = self.action_gradients             
                       
            #initialize all tensor variable parameters:
            self.sess.run(tf.global_variables_initializer())
            
            #To make sure critic and target have same parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
				self.t_W1_c.assign(self.W1_c),
				self.t_B1_c.assign(self.B1_c),
				self.t_W2_c.assign(self.W2_c),
				self.t_W2_action_c.assign(self.W2_action_c),
				self.t_B2_c.assign(self.B2_c),
				self.t_W3_c.assign(self.W3_c),
				self.t_B3_c.assign(self.B3_c)
			])
            
            self.update_target_critic_op = [
                self.t_W1_c.assign(TAU*self.W1_c+(1-TAU)*self.t_W1_c),
                self.t_B1_c.assign(TAU*self.B1_c+(1-TAU)*self.t_B1_c),
                self.t_W2_c.assign(TAU*self.W2_c+(1-TAU)*self.t_W2_c),
                self.t_W2_action_c.assign(TAU*self.W2_action_c+(1-TAU)*self.t_W2_action_c),
                self.t_B2_c.assign(TAU*self.B2_c+(1-TAU)*self.t_B2_c),
                self.t_W3_c.assign(TAU*self.W3_c+(1-TAU)*self.t_W3_c),
                self.t_B3_c.assign(TAU*self.B3_c+(1-TAU)*self.t_B3_c)
            ]
            
    def create_critic_net(self, num_states=4, num_actions=1):
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        critic_state_in = tf.placeholder("float",[None,num_states])
        critic_action_in = tf.placeholder("float",[None,num_actions])    
    
        W1_c = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        B1_c = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        W2_c = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))    
        W2_action_c = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))    
        B2_c= tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions))) 
        W3_c= tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003))
        B3_c= tf.Variable(tf.random_uniform([1],-0.003,0.003))
    
        H1_c=tf.nn.softplus(tf.matmul(critic_state_in,W1_c)+B1_c)
        H2_c=tf.nn.tanh(tf.matmul(H1_c,W2_c)+tf.matmul(critic_action_in,W2_action_c)+B2_c)
            
        critic_q_model=tf.matmul(H2_c,W3_c)+B3_c
            
       
        return W1_c, B1_c, W2_c, W2_action_c, B2_c, W3_c, B3_c, critic_q_model, critic_state_in, critic_action_in
    
    def train_critic(self, state_t_batch, action_batch, y_i_batch ):
        self.sess.run(self.optimizer, feed_dict={self.critic_state_in: state_t_batch, self.critic_action_in:action_batch, self.q_value_in: y_i_batch})
             
    
    def evaluate_target_critic(self,state_t_1,action_t_1):
        return self.sess.run(self.t_critic_q_model, feed_dict={self.t_critic_state_in: state_t_1, self.t_critic_action_in: action_t_1})    
        
    def compute_delQ_a(self,state_t,action_t):
        """
        print ('\n')
        print('check grad number'   )

        ch, act_grad_v= self.sess.run([self.check_fl, self.act_grad_v], feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t})
        print(act_grad_v)

        print(ch)
        print(len(ch))
        print (len(ch[0]) )
        input('Press Enter to continue...')
        """
        return self.sess.run(self.action_gradients, feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t})

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)



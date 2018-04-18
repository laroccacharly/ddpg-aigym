import pdb

import numpy as np
import tensorflow as tf
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
torch.manual_seed(0)
tf.set_random_seed(1)
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TAU = 0.001
ACTOR_LR = LEARNING_RATE

INIT_PARAM_MIN = -3e-3
INIT_PARAM_MAX = 3e-3
HIDDEN_SIZE_1 = 400
HIDDEN_SIZE_2 = 300

def np_to_var(x):
    x = np.array(x)
    x = torch.Tensor(x)

    if torch.cuda.is_available():
        x.cuda()

    return Variable(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        if torch.cuda.is_available():
            print('Gpu active')
            self.cuda()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_to_numpy(self, *args):
        return self.forward(*args).data.cpu().numpy()

    def init_params_from_model(self, model):
        self.load_state_dict(model.state_dict())

    def update_params_from_model(self, model, tau):
        for param1, param2 in zip(self.parameters(), model.parameters()):
            param1.data = tau * param2.data + (1 - tau) * param1.data


class ActorNetPy(Net):
    def __init__(self, nb_features, nb_hidden_1, nb_hidden_2, learning_rate, batch_norm=False):
        super(ActorNetPy, self).__init__()

        self.relu = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm1d(nb_features)
        self.linear_1 = nn.Linear(nb_features, nb_hidden_1)
        #self.linear_1.weight.data.uniform_(INIT_PARAM_MIN, INIT_PARAM_MAX)
        self.batch_norm_2 = nn.BatchNorm1d(nb_hidden_1)
        self.linear_2 = nn.Linear(nb_hidden_1, nb_hidden_2)
        self.batch_norm_3 = nn.BatchNorm1d(nb_hidden_2)
        self.linear_3 = nn.Linear(nb_hidden_2, 1)
        self.tanh = nn.Tanh()
        self.soft = nn.Softplus()

        if batch_norm:
            self.layers = [
                self.batch_norm_1,
                self.linear_1,
                self.relu,
                self.batch_norm_2,
                self.linear_2,
                self.relu,
                self.batch_norm_3,
                self.linear_3,
                self.tanh
            ]
        else:
            self.layers = [
                self.linear_1,
                self.soft,
                self.linear_2,
                self.tanh,
                self.linear_3,
                #self.tanh
            ]


        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)



def weight_init(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform(m.weight)
        m.weight.data.uniform_(INIT_PARAM_MIN, INIT_PARAM_MAX)


class ActorNet:
    """ Actor Network Model of DDPG Algorithm """
    
    def __init__(self,num_states, num_actions):


        self.actor = ActorNetPy(nb_features=num_states, nb_hidden_1=HIDDEN_SIZE_1, nb_hidden_2=HIDDEN_SIZE_2,
                         learning_rate=ACTOR_LR)
        self.actor.apply(weight_init)
        self.target_actor = ActorNetPy(nb_features=num_states, nb_hidden_1=HIDDEN_SIZE_1, nb_hidden_2=HIDDEN_SIZE_2, learning_rate=ACTOR_LR)
        #copy.deepcopy(self.actor)

        self.target_actor.init_params_from_model(self.actor)
        
    def evaluate_actor(self, state):
        state = np_to_var(state)
        return self.actor.forward_to_numpy(state)
        
    def evaluate_target_actor(self, state):
        state = np_to_var(state)
        return self.target_actor.forward_to_numpy(state)

    def train_actor(self, state, q_gradient_input):
        state = np_to_var(state)
        q_gradient_input = np_to_var(q_gradient_input)

        actions = self.actor(state)
        actions.backward(-q_gradient_input)
        self.actor.opt.step()
        self.actor.opt.zero_grad()
        return

    def update_target_actor(self):
        self.target_actor.update_params_from_model(self.actor, TAU)


class ActorNetOld:
    """ Actor Network Model of DDPG Algorithm """

    def __init__(self, num_states, num_actions):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            # actor network model parameters:
            self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a, \
            self.actor_state_in, self.actor_model = self.create_actor_net(num_states, num_actions)

            # target actor network model parameters:
            self.t_W1_a, self.t_B1_a, self.t_W2_a, self.t_B2_a, self.t_W3_a, self.t_B3_a, \
            self.t_actor_state_in, self.t_actor_model = self.create_actor_net(num_states, num_actions)

            # cost of actor network:
            self.q_gradient_input = tf.placeholder("float", [None,
                                                             num_actions])  # gets input from action_gradient computed in critic network file
            self.actor_parameters = [self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a]
            self.parameters_gradients = tf.gradients(self.actor_model, self.actor_parameters,
                                                     -self.q_gradient_input)  # /BATCH_SIZE)
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
                zip(self.parameters_gradients, self.actor_parameters))
            # initialize all tensor variable parameters:
            self.sess.run(tf.global_variables_initializer())

            # To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_W1_a.assign(self.W1_a),
                self.t_B1_a.assign(self.B1_a),
                self.t_W2_a.assign(self.W2_a),
                self.t_B2_a.assign(self.B2_a),
                self.t_W3_a.assign(self.W3_a),
                self.t_B3_a.assign(self.B3_a)])

            self.update_target_actor_op = [
                self.t_W1_a.assign(TAU * self.W1_a + (1 - TAU) * self.t_W1_a),
                self.t_B1_a.assign(TAU * self.B1_a + (1 - TAU) * self.t_B1_a),
                self.t_W2_a.assign(TAU * self.W2_a + (1 - TAU) * self.t_W2_a),
                self.t_B2_a.assign(TAU * self.B2_a + (1 - TAU) * self.t_B2_a),
                self.t_W3_a.assign(TAU * self.W3_a + (1 - TAU) * self.t_W3_a),
                self.t_B3_a.assign(TAU * self.B3_a + (1 - TAU) * self.t_B3_a)]

    def create_actor_net(self, num_states=4, num_actions=1):
        """ Network that takes states and return action """
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        actor_state_in = tf.placeholder("float", [None, num_states])
        W1_a = tf.Variable(
            tf.random_uniform([num_states, N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        B1_a = tf.Variable(tf.random_uniform([N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        W2_a = tf.Variable(
            tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)))
        B2_a = tf.Variable(tf.random_uniform([N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)))
        W3_a = tf.Variable(tf.random_uniform([N_HIDDEN_2, num_actions], -0.003, 0.003))
        B3_a = tf.Variable(tf.random_uniform([num_actions], -0.003, 0.003))

        H1_a = tf.nn.softplus(tf.matmul(actor_state_in, W1_a) + B1_a)
        H2_a = tf.nn.tanh(tf.matmul(H1_a, W2_a) + B2_a)
        actor_model = tf.matmul(H2_a, W3_a) + B3_a
        return W1_a, B1_a, W2_a, B2_a, W3_a, B3_a, actor_state_in, actor_model

    def evaluate_actor(self, state_t):
        return self.sess.run(self.actor_model, feed_dict={self.actor_state_in: state_t})

    def evaluate_target_actor(self, state_t_1):
        return self.sess.run(self.t_actor_model, feed_dict={self.t_actor_state_in: state_t_1})

    def train_actor(self, actor_state_in, q_gradient_input):
        self.sess.run(self.optimizer,
                      feed_dict={self.actor_state_in: actor_state_in, self.q_gradient_input: q_gradient_input})


    def update_target_actor(self):
        self.sess.run(self.update_target_actor_op)


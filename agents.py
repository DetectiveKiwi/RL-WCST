import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Naive_Tabular_Agent():
    def __init__(self, epsilon, alpha, gamma, lam, num_cards):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.num_cards = num_cards
        self.Q = {}
        self.E = {}
        
        self.trial_num = 0
        self.r = 0
        self.r2 = 0
        self.a = 0
        self.a2 = 0
        self.s = ''
        self.s2 = ''

    def choose_action(self, cards):
        self.s2 = str(list(cards.reshape((-1))))
        
        for a in range(self.num_cards):
            key = self.s2 + str(a)
            if key not in self.Q:
                self.Q[key] = 0.
                self.E[key] = 0.
        
        if np.random.random() > self.epsilon:
            q_val = np.empty((self.num_cards))
            for a in range(self.num_cards):
                key = self.s2 + str(a)
                q_val[a] = self.Q[key]
                    
            self.a2 = np.random.choice(np.argwhere(q_val==np.max(q_val))[:,0])
        else:
            self.a2 = np.random.choice(self.num_cards)

        return self.a2
    
    def update(self, reward):
        self.r2 = reward
        if self.trial_num > 0:
            key = self.s + str(self.a)
            key2 = self.s2 + str(self.a2)
            delta = self.r + self.gamma * self.Q[key2] - self.Q[key]
            self.E[key] = self.E[key] + 1
            
            for key in self.Q:
                self.Q[key] = self.Q[key] + self.alpha * delta * self.E[key]
                self.E[key] = self.gamma * self.lam * self.E[key]
            
        self.s = self.s2
        self.a = self.a2
        self.r = self.r2
        self.trial_num += 1
        
        
class Feature_Tabular_Agent():
    def __init__(self, epsilon, alpha, gamma, lam, num_cards):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.num_cards = num_cards
        self.Q = {}
        self.E = {}
        
        self.trial_num = 0
        self.r = 0
        self.r2 = 0
        self.a = 0
        self.a2 = 0
        self.s = ''
        self.s2 = ''
        self.choice = np.array([-1,-1,-1])
        self.choice2 = np.array([-1,-1,-1])
        
        action_space = []
        for i in range(0,4):
            action_space.append([i,-1,-1])
        for j in range(4,8):
            action_space.append([-1,j,-1])
        for k in range(8,12):
            action_space.append([-1,-1,k])
        self.action_space = np.array(action_space)
        
        for a in range(len(self.action_space)):
            key = self.s2 + str(a)
            if key not in self.Q:
                self.Q[key] = 0.
                self.E[key] = 0.

    def choose_action(self, cards):
        self.s2 = ''
        
        if np.random.random() > self.epsilon:
            q_val = np.empty((len(self.action_space)))
            for a in range(len(self.action_space)):
                key = self.s2 + str(a)
                temp = np.argwhere(np.any(cards==self.action_space[a], axis=1))[:,0]
                if len(temp)==1:
                    q_val[a] = self.Q[key]
                else:
                    q_val[a] = -np.inf
                    
            self.a2 = np.random.choice(np.argwhere(q_val==np.max(q_val))[:,0])
        else:
            q_val = np.empty((len(self.action_space)))
            for a in range(len(self.action_space)):
                temp = np.argwhere(np.any(cards==self.action_space[a], axis=1))[:,0]
                if len(temp)==1:
                    q_val[a] = 0
                else:
                    q_val[a] = -np.inf
                self.a2 = np.random.choice(len(self.action_space))
                
        temp = np.argwhere(np.any(cards==self.action_space[self.a2], axis=1))[0,0]
        self.choice2 = cards[temp]

        return temp
    
    def update(self, reward):
        self.r2 = reward
        if self.trial_num > 0:
            key2 = self.s2 + str(self.a2)
            actions = []
            actions.append(np.argwhere(np.sum(self.choice==self.action_space[:12], axis=1)==1)[:,0])
            actions = np.hstack(actions)
            for a in actions:
                key = self.s + str(a)
                delta = self.r + self.gamma * self.Q[key2] - self.Q[key]
                self.E[key] = self.E[key] + 1

                for key in self.Q:
                    self.Q[key] = self.Q[key] + self.alpha * delta * self.E[key]
                    self.E[key] = self.gamma * self.lam * self.E[key]
            
        self.s = self.s2
        self.a = self.a2
        self.r = self.r2
        self.choice = self.choice2
        self.trial_num += 1
        
        
        
class two_layer_network(nn.Module):
    def __init__(self, lr, input_size, hidden_size):
        super(two_layer_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ih = nn.Linear(self.input_size, self.hidden_size)
        self.ho = nn.Linear(self.hidden_size,1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
    
    def forward(self, input_state):
        hidden = self.ih(input_state)
        output = self.ho(hidden)
        return output

class NN_Agent():
    def __init__(self, epsilon, alpha, num_cards, num_dims):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_cards = num_cards
        self.num_dims = num_dims
        
        self.Q_eval = two_layer_network(self.alpha, self.num_cards ** self.num_dims, self.num_cards * self.num_dims)
        
        self.stimuli = {}
        count = 0
        for i in range(0,4):
            for j in range(4,8):
                for k in range(8,12):
                    key = str(np.array([i,j,k]))
                    self.stimuli[key] = count
                    count += 1
                    
        self.choice = -1

    def choose_action(self, cards):
        q_val = np.empty(self.num_cards)
        for i,card in enumerate(cards):
            input_state = T.zeros((self.num_cards ** self.num_dims))
            input_state[self.stimuli[str(card)]] = 1
            q_val[i] = self.Q_eval.forward(input_state)
            
        if np.random.random() > self.epsilon:
            action = np.random.choice(np.argwhere(q_val==np.max(q_val))[:,0])
        else:
            action = np.random.choice(self.num_cards)
            
        self.choice = self.stimuli[str(cards[action])]

        return action
    
    def update(self, reward):
        self.Q_eval.optimizer.zero_grad()
        
        input_state = T.zeros((self.num_cards ** self.num_dims))
        input_state[self.choice] = 1
        q_val = self.Q_eval.forward(input_state)
        
        loss = self.Q_eval.loss(q_val, T.tensor([reward], dtype=T.float))
        loss.backward()
        self.Q_eval.optimizer.step()
        
        
        
class feature_rnn_network(nn.Module):
    def __init__(self, lr, weight_decay, input_size, hidden_size):
        super(feature_rnn_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay

        self.hh = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        self.loss = nn.MSELoss()
    
    def forward(self, features, hidden_state):
        # Features is 3 hot out of 12 for this card
        output = T.sum((features * hidden_state)[None,:], dim=1)
        return output
    
    def update_hidden(self, input_state, hidden_state, reward):
        # input_state is one hot of 64 stimuli
        hh1 = T.cat([input_state*reward, hidden_state])
        hh2 = self.hh(hh1)
        return hh2
    
    def init_hidden(self):
        return T.zeros(self.hidden_size)
    
class feature_rnn_fb_network(nn.Module):
    def __init__(self, lr, weight_decay, input_size, hidden_size):
        super(feature_rnn_fb_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay

        self.hh_cor = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        self.hh_inc = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        self.loss = nn.MSELoss()
    
    def forward(self, features, hidden_state):
        output = T.sum((features * hidden_state)[None,:], dim=1)
        return output
    
    def update_hidden(self, input_state, hidden_state, reward):
        hh1 = T.cat([input_state, hidden_state])
        if reward:
            hh2 = self.hh_cor(hh1)
        else:
            hh2 = self.hh_inc(hh1)
        return hh2
    
    def init_hidden(self):
        return T.zeros(self.hidden_size)

class RNN_Feature_Agent():
    def __init__(self, epsilon, alpha, weight_decay, num_cards, num_dims, separate_fb=False):
        self.epsilon = epsilon
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.num_cards = num_cards
        self.num_dims = num_dims
        
        self.feature = {}
        count = 0
        for i in range(0,4):
            for j in range(4,8):
                for k in range(8,12):
                    key = str(np.array([i,j,k]))
                    self.feature[key] = count
                    count += 1

        if separate_fb:
            self.Q_eval = feature_rnn_fb_network(self.alpha, self.weight_decay, len(self.feature), num_cards*num_dims)
        else:
            self.Q_eval = feature_rnn_network(self.alpha, self.weight_decay, len(self.feature), num_cards*num_dims)
        
        self.choice = np.array([-1,-1,-1])
        self.prev_choice = self.choice.copy()
        self.hidden = self.Q_eval.init_hidden()
        self.prev_hidden = self.hidden.detach()
            
    def choose_action(self, cards):   
        q_val = np.zeros(self.num_cards)
        for i,card in enumerate(cards):
            features = T.zeros((self.num_cards*self.num_dims))
            features[card] = 1
            q_val[i] = self.Q_eval.forward(features, self.hidden)
            
        if np.random.random() > self.epsilon:
            action = np.random.choice(np.argwhere(q_val==np.max(q_val))[:,0])
        else:
            action = np.random.choice(self.num_cards)
            
        self.prev_choice = self.choice.copy()
        self.choice = cards[action]

        return action

    def update_hidden(self, reward):
        reward_state = T.tensor([reward], dtype=T.float)  
        input_state = T.zeros((len(self.feature)))
        input_state[self.feature[str(self.choice)]] = 1
        
        temp = self.Q_eval.update_hidden(input_state, self.hidden, reward_state)
        
        self.prev_hidden = self.hidden.detach()
        self.hidden = temp.detach()
        
    def learn_hidden_update(self, prev_reward, reward):
        self.Q_eval.optimizer.zero_grad()
        prev_reward_state = T.tensor([prev_reward], dtype=T.float)
        reward_state = T.tensor([reward], dtype=T.float)
        
        prev_input_state = T.zeros((len(self.feature)))
        prev_input_state[self.feature[str(self.prev_choice)]] = 1
        
        features = T.zeros((self.num_cards*self.num_dims))
        features[self.choice] = 1
            
        hidden_new = self.Q_eval.update_hidden(prev_input_state, self.prev_hidden, prev_reward_state)
        q_val = self.Q_eval.forward(features, hidden_new)

        loss = self.Q_eval.loss(q_val, reward_state)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        
            
class stimulus_rnn_fb_network(nn.Module):
    def __init__(self, lr, weight_decay, input_size, hidden_size):
        super(stimulus_rnn_fb_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay

        self.hh_cor = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        self.hh_inc = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        self.loss = nn.MSELoss()
    
    def forward(self, input_state, hidden_state):
        output = T.sum((input_state * hidden_state)[None,:], dim=1)
        return output
    
    def update_hidden(self, input_state, hidden_state, reward):
        hh1 = T.cat([input_state, hidden_state])
        if reward:
            hh2 = self.hh_cor(hh1)
        else:
            hh2 = self.hh_inc(hh1)
        return hh2
    
    def init_hidden(self):
        return T.zeros(self.hidden_size)

class RNN_Stimulus_Agent():
    def __init__(self, epsilon, alpha, weight_decay, num_cards, num_dims):
        self.epsilon = epsilon
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.num_cards = num_cards
        self.num_dims = num_dims
        
        self.feature = {}
        count = 0
        for i in range(0,4):
            for j in range(4,8):
                for k in range(8,12):
                    key = str(np.array([i,j,k]))
                    self.feature[key] = count
                    count += 1

        self.Q_eval = stimulus_rnn_fb_network(self.alpha, self.weight_decay, len(self.feature), len(self.feature))
        
        self.choice = np.array([-1,-1,-1])
        self.prev_choice = self.choice.copy()
        self.hidden = self.Q_eval.init_hidden()
        self.prev_hidden = self.hidden.detach()
            
    def choose_action(self, cards):   
        q_val = np.zeros(self.num_cards)
        for i,card in enumerate(cards):
            input_state = T.zeros((len(self.feature)))
            input_state[self.feature[str(card)]] = 1
            q_val[i] = self.Q_eval.forward(input_state, self.hidden)
            
        if np.random.random() > self.epsilon:
            action = np.random.choice(np.argwhere(q_val==np.max(q_val))[:,0])
        else:
            action = np.random.choice(self.num_cards)
            
        self.prev_choice = self.choice.copy()
        self.choice = cards[action]

        return action

    def update_hidden(self, reward):
        reward_state = T.tensor([reward], dtype=T.float)  
        input_state = T.zeros((len(self.feature)))
        input_state[self.feature[str(self.choice)]] = 1
        
        temp = self.Q_eval.update_hidden(input_state, self.hidden, reward_state)
        
        self.prev_hidden = self.hidden.detach()
        self.hidden = temp.detach()
        
    def learn_hidden_update(self, prev_reward, reward):
        self.Q_eval.optimizer.zero_grad()
        prev_reward_state = T.tensor([prev_reward], dtype=T.float)
        reward_state = T.tensor([reward], dtype=T.float)
        
        prev_input_state = T.zeros((len(self.feature)))
        prev_input_state[self.feature[str(self.prev_choice)]] = 1
        
        input_state = T.zeros((len(self.feature)))
        input_state[self.feature[str(self.choice)]] = 1
            
        hidden_new = self.Q_eval.update_hidden(prev_input_state, self.prev_hidden, prev_reward_state)
        q_val = self.Q_eval.forward(input_state, hidden_new)

        loss = self.Q_eval.loss(q_val, reward_state)
        loss.backward()
        self.Q_eval.optimizer.step()
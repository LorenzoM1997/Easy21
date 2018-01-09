import random
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def hit():
    number = random.randint(1,10)
    color = random.randint(1,3)
    if color == 1:
        #red card
        number = -number
    return number

def step(state, action):
    player = state[0]
    dealer = state[1]
    reward = 0
    state_ = (player, dealer)
    if action == 0:
        #stick
        while dealer < 17:
            dealer += hit()
            if dealer < 1:
                break
        if dealer > 21 or dealer < 1:
            state_ = "terminal"
            reward = 1
        else:
            if player > dealer:
                state_ = "terminal"
                reward = 1
            elif player < dealer:
                state_ = "terminal"
                reward = -1
            else:
                state_ = "terminal"
    else:
        #hit
        player += hit()
        if player > 21 or player < 1:
            state_ = "terminal"
            reward = -1  
    if state_ is not "terminal":
        state_ = (player, dealer)
    return state_, reward


"""
MONTE CARLO CONTROL
"""

Q = np.zeros((10*21, 2))
N = np.zeros((10*21, 2))
n0 = 100

n_episodes = 500000

for _ in range(n_episodes):
    player = random.randint(1,10)
    dealer = random.randint(1,10)
    state = (player, dealer)
    reward = 0
    visited = np.zeros((0,2), dtype = int)

    while state is not "terminal":
        i_state = ((player - 1) * 10 )+ (dealer - 1)
        
        N_s = np.sum(N[i_state])    
        
        epsilon = n0 / (n0 + N_s)
        if random.random() < epsilon:
            # random action
            action = random.randint(0,1)
        else:
            # greedy action
            action = np.argmax(Q[i_state])

        #since you visit this state, increase its value by 1
        N[i_state][action] += 1
        visited = np.append(visited, [[i_state, action]], axis = 0)
        state, reward = step(state, action)
        player = state[0]
        dealer = state[1]   
    
    # when you have the final reward, update all the states you visited
    for v in visited:
        s = v[0]
        a = v[1]
        if N[s][a] != 0:
            Q[s][a] = Q[s][a] + (1 / N[s][a]) * (reward - Q[s][a])

# make a graph   
xs = np.zeros(210, dtype = int)
for x in range(210):
	xs[x] = np.floor(x/10)
xs = xs +1
ys = np.zeros(210, dtype = int)
for y in range(210):
    ys[y] = y % 10
ys = ys + 1
V = np.max(Q, axis = 1)
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(xs,ys,V)
pyplot.show()

"""
TD LEARNING
"""

Q = np.zeros((10*21, 2))
E = np.zeros((10*21, 2))
n0 = 100

for _ in range(1):
    lambd = 0.9
    episodes = 100000
    for episode in range(episodes):
        
        player = random.randint(1,10)
        dealer = random.randint(1,10)
        state = (player, dealer)

        # epsilon greedy choice of action
        i_state = ((player - 1) * 10 )+ (dealer - 1)  
        epsilon = 1
        if random.random() < epsilon:
            # random action
            action = random.randint(0,1)
        else:
            # greedy action
            action = np.argmax(Q[i_state])

        # Repeat (for each step of the episode)
        while state is not "terminal":
            i_state = ((state[0] - 1) * 10 )+ (state[1] - 1)
            # Take action A, observe R, S'
            state_, reward_ = step(state, action)
            E[i_state][action] += 1
            N[i_state][action] += 1
            alpha = 1/N[i_state][action]
            
            if state_ is not "terminal":
                # Choose A' from S' using policy derived from Q( epsilon greedy)
                i_state_ = ((state_[0] - 1) * 10 )+ (state_[1] - 1)
                epsilon = 1 / (episode + 1)
                if random.random() < epsilon:
                    # random action
                    action_ = random.randint(0,1)
                else:
                    # greedy action
                    action_ = np.argmax(Q[i_state])

                delta = reward_ + Q[i_state_][action_] - Q[i_state][action]
                action = action_
                
            else:
                delta = reward_ - Q[i_state][action]            

            Q = Q + alpha * delta * E
            E = lambd * E
            
            state = state_

# make a graph   
xs = np.zeros(210, dtype = int)
for x in range(210):
    xs[x] = np.floor(x/10)
xs = xs +1
ys = np.zeros(210, dtype = int)
for y in range(210):
    ys[y] = y % 10
ys = ys + 1
V = np.max(Q, axis = 1)
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(xs,ys,V)
pyplot.show()

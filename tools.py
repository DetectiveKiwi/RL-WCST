import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from tqdm import tqdm

def moving_average(a, n):
    """
    Calculates the moving average of an array.
    Function taken from Jaime here:
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    
    Parameters
    ----------
    a: array to be averaged
    n: size of window
    
    Returns
    --------------
    Moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def spherical_2_cartesian(r, theta, phi):
    '''
    Coordinate transformation, from spherical to cartesian
    
    Parameters
    ----------
    r: radius
    theta: theta
    phi: phi
    
    Returns
    -------
    x: x position
    y: y position
    z: z position
    '''
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    return x, y, z

def train_agent(sess, agent, game_len):
    '''
    Runs and trains the agent
    
    Parameters
    ----------
    sess: wcst session instance
    agent: agent to train
    game_len: number of trials
    
    Returns
    -------
    sess: the new wcst session instance
    agent: the trained agent
    scores: the achieved scores
    '''
    scores = []
    sess.start_new_session()

    for j in tqdm(range(game_len)):
        cards = sess.get_cards()

        idx = np.argsort(cards[:,0])
        cards_ordered = cards[idx]

        action_ordered = agent.choose_action(cards_ordered)
        action = idx[action_ordered]

        feedback = sess.make_selection(action)
        reward = feedback * 2 - 1

        agent.update(reward)

        scores.append(feedback)
        
    return sess, agent, scores

def train_rnn_agent(sess, agent, game_len, learning_rate, learning_schedule, restart_game=True):
    '''
    Runs and trains the RNN agent, which updates the hidden state update
    
    Parameters
    ----------
    sess: wcst session instance
    agent: agent to train
    game_len: number of trials
    learning_rate: list comprising of learning rates (number of trials between learning)
    learning_schedule: list comprising trial number when learning rate is updated
    restart_game: Flag to say whether or not to restart the game every 100000 trials
    
    Returns
    -------
    sess: the new wcst session instance
    agent: the trained agent
    scores: the achieved scores
    '''
    num = 0
    reward = 0
    scores = []
    
    sess.start_new_session()

    for j in tqdm(range(game_len)):
        if restart_game and j%100000==0:
            sess.start_new_session()
        cards = sess.get_cards()

        action = agent.choose_action(cards)

        prev_reward = reward
        feedback = sess.make_selection(action)
        reward = feedback * 2 - 1

        if j>1:
            if j%learning_rate[num]==0:
                agent.learn_hidden_update(prev_reward, reward)

        agent.update_hidden(reward)

        scores.append(feedback)
        if j==learning_schedule[num]:
            num = num + 1
        
    return sess, agent, scores

def run_rnn_agent(sess, agent, game_len):
    '''
    Runs the RNN agent, without any learning
    
    Parameters
    ----------
    sess: wcst session instance
    agent: agent to train
    game_len: number of trials
    
    Returns
    -------
    sess: the new wcst session instance
    scores: the achieved scores
    '''
    scores = []
    
    sess.start_new_session()

    for j in tqdm(range(game_len)):
        cards = sess.get_cards()

        action = agent.choose_action(cards)

        feedback = sess.make_selection(action)
        reward = feedback * 2 - 1

        agent.update_hidden(reward)

        scores.append(feedback)
        
    return sess, scores

def plot_scores(scores, num_to_avg, ideal_mean, ideal_std, title=''):
    '''
    Plots the scores after applying a moving average
    
    Parameters
    ----------
    scores: the achieved scores
    num_to_avg: the number of trials to average over
    ideal_mean: ideal agents mean performance
    ideal_std: ideal agents standard deviation of performance
    title: Title to include, if desired
    '''
    plt.plot(moving_average(scores,num_to_avg))
    plt.axhline(1. / 4, color='black', linestyle='dashed')
    plt.axhline(ideal_mean, color='black', linestyle='dashed')
    plt.fill_between(
        np.arange(len(scores)),
        ideal_mean-ideal_std,
        ideal_mean+ideal_std,
        color='black',
        alpha=0.2
    )
    plt.xlim(0,len(scores))
    plt.xlabel('Trial Number')
    plt.ylabel('Percent Correct Over '+str(num_to_avg)+' Trials')
    plt.title(title)
    
def average_stim(stim):
    '''
    Averages stimulus based representation across all inputs, 
        re-ordered according to Figures/rolling.png
        
    Parameters
    ----------
    stim : 64 by 64 matrix, where each array (second dim) is re-ordered
    
    Returns
    -------
    mean_stim : stim, averaged across first second dimension
    '''
    x = []
    y = []
    z = []

    for i in range(4):
        for j in range(4):
            for k in range(4):
                x.append(i)
                y.append(j)
                z.append(k)

    coords = pd.DataFrame()
    coords_all = []

    count = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                coords = pd.DataFrame()
                coords['x'] = np.roll(x, 16*i)
                coords['y'] = np.roll(y, 4*j)
                coords['z'] = np.roll(z, k)
                coords['stim'] = stim[count]
                coords['count'] = np.ones(len(x), dtype=int) * count
                coords_all.append(coords)
                count += 1

    coords = pd.concat(coords_all, ignore_index=True)
    mean_stim = []

    for i in range(len(x)):
        mean_stim.append(
            np.mean(coords[(coords['x']==x[i]) & (coords['y']==y[i]) & (coords['z']==z[i])].stim.values)
        )
    
    return np.array(mean_stim)

def plot_stim_by_dim(stim, stim_name='amp', color_lim=0.24, r=2.5, theta=-np.pi/4+0.4, phi=np.pi/2-0.05):
    '''
    Plots the stimulus representation by feature dimension
    
    Parameters
    ----------
    stim : length 64 array representing the values, ordered by dimensions
    stim_name : name of the stimulus representation
    color_lim : color limit, from -color_lim to +color_lim
    r : radial component of camera view in spherical coordinates
    theta : theta component of camera view in spherical coordinates
    phi: phi component of camera view in spherical coordinates
    '''
    x = []
    y = []
    z = []

    for i in range(4):
        for j in range(4):
            for k in range(4):
                x.append(i)
                y.append(j)
                z.append(k)

    coords = pd.DataFrame()
    coords['dim_1'] = x
    coords['dim_2'] = y
    coords['dim_3'] = z
    coords[stim_name] = stim

    fig = px.scatter_3d(
        coords, 
        x='dim_1',
        y='dim_2',
        z='dim_3',
        color=stim_name,
        range_color=[-color_lim,color_lim], 
        color_continuous_scale='RdBu'
    )
    x, y, z = spherical_2_cartesian(r, theta, phi)
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=x, y=y, z=z)
    )
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(tickvals=np.arange(4)),
            yaxis=dict(tickvals=np.arange(4)),
            zaxis=dict(tickvals=np.arange(4)),
        )
    )
    fig.show('png')
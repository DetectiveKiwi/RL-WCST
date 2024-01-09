import numpy as np
import matplotlib.pyplot as plt
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
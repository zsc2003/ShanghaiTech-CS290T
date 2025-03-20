# Episodic Model Based Learning using Maximum Likelihood Estimate of the Environment
# Do not change the arguments and output types of any of the functions provided!
import numpy as np
import gym
import copy

from dynamic_programming import ValueIteration, print_agent

import matplotlib.pyplot as plt


def initialize_P(nS, nA):
    """
    Initializes a uniformly random model of the environment with 0 rewards.

    Parameters
    ----------
    nS: int
        Number of states
    nA: int
        Number of actions

    Returns
    -------
    P: np.array of shape [nS x nA x nS x 4] where items are tuples representing transition information
        P[state][action] is a list of (prob, next_state, reward, done) tuples.
    """

    P = [[[(1.0 / nS, i, 0, False) for i in range(nS)] for _ in range(nA)] for _ in range(nS)]
    return P


def initialize_counts(nS, nA):
    """
    Initializes a counts array.

    Parameters
    ----------
    nS: int
        Number of states
    nA: int
        Number of actions

    Returns
    -------
    counts: np.array of shape [nS x nA x nS]
        counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
    """

    counts = [[[0 for _ in range(nS)] for _ in range(nA)] for _ in range(nS)]
    return counts


def initialize_rewards(nS, nA):
    """
    Initializes a rewards array. Values represent running averages.

    Parameters
    ----------
    nS: int
        Number of states
    nA: int
        Number of actions

    Returns
    -------
    rewards: array of shape [nS x nA x nS]
        counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"
    """

    rewards = [[[0 for _ in range (nS)] for _ in range(nA)] for _ in range(nS)]
    return rewards


def counts_and_rewards_to_P(counts, rewards):
    """
    Converts counts and rewards arrays to a P array consistent with the Gym environment data structure for a model of the environment.
    Use this function to convert your counts and rewards arrays to a P that you can use in value iteration.

    Parameters
    ----------
    counts: array of shape [nS x nA x nS]
        counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
    rewards: array of shape [nS x nA x nS]
        counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"

    Returns
    -------
    P: np.array of shape [nS x nA x nS x 4] where items are tuples representing transition information
        P[state][action] is a list of (prob, next_state, reward, done) tuples.
    """
    nS = len(counts)
    nA = len(counts[0])
    P = [[[] for _ in range(nA)] for _ in range(nS)]


    # for s in range(env.nS):
    #     for a in range(env.nA):
    #         for s_new in range(env.nS):
    #             # (prob, next_state, reward, done)
    #             if counts[s][0][s] == -1: # s is the terminal state
    #                 P[s][a][s_new] = (1, s, 0, True)
    #             elif counts[s_new][a][s_new] == -1: # s_new is the terminal state
    #                 P[s][a][s_new] = (counts[s][a][s_new] / sum(counts[s][a]), s_new, rewards[s][a][s_new], True)
    #             else:
    #                 P[s][a][s_new] = (counts[s][a][s_new] / sum(counts[s][a]), s_new, rewards[s][a][s_new], False)

    for state in range(nS):
        for action in range(nA):
            if sum(counts[state][action]) > 0:
                for next_state in range(nS):
                    if counts[state][action][next_state] != 0:
                        prob = float(counts[state][action][next_state]) / float(sum(counts[state][action]))
                        reward = rewards[state][action][next_state]
                        P[state][action].append((prob, next_state, reward, False))
            elif sum(counts[state][action]) == 0:
                prob = 1.0 / float(nS)
                for next_state in range(nS):
                    P[state][action].append((prob, next_state, 0, False))
            else: # terminal state
                P[state][action].append((1.0, state, 0, True))

    return P


def update_mdp_model_with_history(counts, rewards, history):
    """
    Given a history of an entire episode, update the count and rewards arrays

    Parameters
    ----------
    counts: array of shape [nS x nA x nS]
        counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
    rewards: array of shape [nS x nA x nS]
        [state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"
    history
        a list of [state, action, reward, next_state, done]
    """

    # HINT: For terminal states, we define that the probability of any action returning
    # the state to itself is 1 (with zero reward)
    # Make sure you record this information in your counts array by updating the counts
    # for this accordingly for your value iteration to work.

    nA = len(counts[0])

    # history: [[state, action, reward, next_state, done], [state, action, reward, next_state, done], ...]
    for state, action, reward, next_state, done in history:
        counts[state][action][next_state] += 1
        # average reward (x+r)/n = x / (n-1) * (n-1 / n) + r / n
        rewards[state][action][next_state] = rewards[state][action][next_state] * (counts[state][action][next_state] - 1) / counts[state][action][next_state] + reward / counts[state][action][next_state]
        if done:
            # next_state is the terminal state
            for a in range(nA):
                counts[next_state][a][next_state] = -10000

def learn_with_mdp_model(env, num_episodes=5000, gamma = 0.95, e = 0.8, decay_rate = 0.99):
    """
    Build a model of the environment and use value iteration to learn a policy. In the next episode, play with the new
    policy using epsilon-greedy exploration.
    (Since we use the colected episodes, we just ignore the epsilon-greedy exploration part)

    Your model of the environment should be based on updating counts and rewards arrays. The counts array counts the number
    of times that "state" with "action" led to "next_state", and the rewards array is the running average of rewards for
    going from at "state" with "action" leading to "next_state".

    For a single episode, create a list called "history" with all the experience
    from that episode, then update the "counts" and "rewards" arrays using the function "update_mdp_model_with_history".

    You may then call the prewritten function "counts_and_rewards_to_P" to convert your counts and rewards arrays to
    an environment data structure P consistent with the Gym environment's one. You may then call on value_iteration(P, nS, nA)
    to get a policy.

    Parameters
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as attributes.
    num_episodes: int
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    e: float
        Epsilon value used in the epsilon-greedy method.
    decay_rate: float
        Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    policy: np.array
        An array of shape [env.nS] representing the action to take at a given state.
    """
    P = initialize_P(env.nS, env.nA)
    counts = initialize_counts(env.nS, env.nA)
    rewards = initialize_rewards(env.nS, env.nA)

    policy = np.zeros(env.nS, dtype=int)
    All_episodes = np.load("All_episodes.npy", allow_pickle=True)

    # After getting MDP, You can directly use ValueIteration in dynamic_programing.py to get policy

    for i in range(min(num_episodes, All_episodes.shape[0])):
        episode = All_episodes[i] # [state, action, reward, next_state, done]
        update_mdp_model_with_history(counts, rewards, episode) # counts, rewards will be modified

    # use counts and rewards to estimate P
    """
    From the document of gym, we can find that:
        Action space:
            0: LEFT
            1: DOWN
            2: RIGHT
            3: UP
        Rewards:
            Reach goal(G): +1
            Reach hole(H): 0
            Reach frozen(F): 0
    """
    P = counts_and_rewards_to_P(counts, rewards)

    agent = ValueIteration(env.ncol, env.nrow, P, theta=0.001, gamma=gamma)
    agent.value_iteration()
    agent.get_policy()

    # for i in range(env.nrow):
    #     print(f'{agent.v[i*env.ncol:(i+1)*env.ncol]}')
    # print(agent.pi)

    policy = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        policy[s] = np.argmax(agent.pi[s])

    return policy


def render_single(env, policy):
    """
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as attributes.
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        # env.render()
        # time.sleep(0.5) # Seconds between frames. Modify as you wish.
        action = policy[state]
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: %f" % episode_reward)
    return episode_reward


# Don't change code here !
def main():
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped
    env.np_random.seed(456465)

    policy = learn_with_mdp_model(env, num_episodes=1000)

    score = []
    for i in range(100):
        episode_reward = render_single(env, policy)
        score.append(episode_reward)
    for i in range(len(score)):
        score[i] = np.mean(score[:i + 1])

    plt.plot(np.arange(100), np.array(score))
    plt.title('The running average score of the model based agent')
    plt.xlabel('training episodes')
    plt.ylabel('score')
    plt.savefig('model_based.png')
    plt.show()


if __name__ == '__main__':
    main()
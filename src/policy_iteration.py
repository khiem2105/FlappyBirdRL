import numpy as np

from copy import deepcopy
from collections import defaultdict

def eval_pol(pi, env, gamma, thresh):
    states = env.P.keys()
    V = dict.fromkeys(states, 0)
    diff = 100000

    while diff > thresh:
        V_old = deepcopy(V)
        for state in states:
            action = pi[state]
            for next in env.P[state][action]:
                prob, next_state, reward = next[:-1]
                V[state] += gamma * prob * V_old[next_state]
            V[state] += reward

        print(list(V.values()))
        print(list(V_old.values()))
        diff = np.max(np.abs(np.array(list(V.values())) - np.array(list(V_old.values()))))

    return V

def get_pol(V, env, gamma):
    pi = dict.fromkeys(V.keys(), None)

    for state in V:
        actions = dict.fromkeys(env.P[state].keys(), 0)
        for action in actions:
            for next in env.P[state][action]:
                prob, next_state, reward = next[:-1]
                actions[action] += gamma * prob * V[next_state]
        actions[action] += reward

        best_action = max(actions.keys(), key=lambda action: actions[action])
        pi[state] = best_action

    return pi

def policy_iteration(env, gamma, thresh):
    n_action = env.action_space.n
    pi = defaultdict(lambda : np.random.randint(n_action))
    V = dict.fromkeys(env.P.keys(), 0)

    diff = 10000

    while diff > thresh:
        V_old = deepcopy(V)
        V = eval_pol(pi, env, gamma, thresh)
        pi = get_pol(V, env, gamma)

        diff = np.max(np.abs(np.array(list(V.values())) - np.array(list(V_old.values()))))

    return pi
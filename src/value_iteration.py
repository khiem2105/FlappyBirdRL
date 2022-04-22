import numpy as np

from copy import deepcopy

from policy_iteration import get_pol

def value_iteration(env, gamma, thresh):
    V = dict.fromkeys(env.P.keys(), 0)

    diff = 10000

    while diff > thresh:
        V_old = deepcopy(V)
        for state in V:
            actions = dict.fromkeys(env.P[state].keys(), 0)
            for action in actions:
                for next in env.P[state][action]:
                    prob, next_state, reward = next[:-1]
                    actions[action] += gamma * prob * V_old[next_state]
                actions[action] += reward
                
            V[state] = max(actions.values())

        diff = np.max(np.abs(np.array(list(V.values())) - np.array(list(V_old.values()))))

    return get_pol(V, env, gamma)
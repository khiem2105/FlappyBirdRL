import gym

from policy_iteration import policy_iteration
from value_iteration import value_iteration

from agents import AgentRandom, AgentPolicy

from play import play_env

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solving Taxi problem by dynamic programming")
    parser.add_argument("--algo", type=str, default="policy_iteration",
                        choices=["policy_iteration", "value_iteration", "random"],
                        help="type of dynamic programming algorithm using")

    parser.add_argument("--gamma", type=float, default=.95,
                        help="Discount factor")

    parser.add_argument("--threshold", type=float, default=.001,
                        help="Convergence threshold")

    args = parser.parse_args()
    algo, gamma, threshold = args.algo, args.gamma, args.threshold

    env_taxi = gym.make("Taxi-v3")

    if algo == "random":
        play_env(AgentRandom(env_taxi))
    elif algo == "policy_iteration":
        print(algo)
        print(gamma)
        print(threshold)
        pi = policy_iteration(env_taxi, gamma, threshold)
        # play_env(AgentPolicy(env_taxi, pi), fps=30)
    else:
        pi = value_iteration(env_taxi, gamma, threshold)
        play_env(AgentPolicy(env_taxi, pi), fps=30)
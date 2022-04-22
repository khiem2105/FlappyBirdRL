import gym
import flappy_bird_gym

import numpy as np

class MyFlappyEnv:
    """ 
    Custom Flappy Env :
    * state : [horizontal delta of the next pipe, vertical delta, vertical velocity]
    """

    def __init__(self):
        self.env = flappy_bird_gym.make('FlappyBird-v0')
        self.env._normalize_obs = False
        self._last_score = 0
    def __getattr__(self,attr):
        return self.env.__getattribute__(attr)
    
    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        if done:
            reward -=1000
        player_x = self.env._game.player_x
        player_y = self.env._game.player_y

        return np.hstack([obs,self.env._game.player_vel_y]),reward, done, info
    def reset(self):
        return np.hstack([self.env.reset(),self.env._game.player_vel_y])
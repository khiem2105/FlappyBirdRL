import gym
import time
import flappy_bird_gym

def test_gym(fps=30):
    env = gym.make('Taxi-v3')
    env.reset()
    r = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        r += reward
    
        env.render()
        time.sleep(1/fps)

        print(f"iter {i}\naction: {action}, reward: {reward}, state: {type(obs)} ")
        
        if done:
            break
    
    print(f"reward cumulatif: {r}")

def test_flappy(fps=30):
    env = flappy_bird_gym.make('FlappyBird-v0')
    env.reset()
    r = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        r += reward

        env.render()
        time.sleep(1/fps)

        print(f"iter {i}\naction: {action}, reward: {reward}, state: {obs}, {info}, {env._game.player_vel_y}")
        
        if done:
            break
        
    print(f"reward cumulatif : {r} ")
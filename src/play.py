import time

def play_env(agent, max_ep=500, fps=-1, verbose=True):
    """
    Play an episode :
    * agent : agent with two functions : act(state) -> action, and store(state,action,state,reward)
    * max_ep : maximal length of the episode
    * fps : frame per second,not rendering if <=0
    * verbose : True/False print debug messages
    * return the cumulative reward
    """
    obs = agent.env.reset()
    cumr = 0

    for i in range(max_ep):
        last_obs = obs
        action = agent.act(obs)
        obs, reward, done, _ = agent.env.step(int(action))
        agent.store(last_obs, action, obs, reward)
        cumr += reward

        if fps>0:
            agent.env.render()
            if verbose: 
                print(f"iter {i}\n{action}: {reward} -> {obs}")   

            time.sleep(1/fps)
        
        if done:
            break
    
    return cumr
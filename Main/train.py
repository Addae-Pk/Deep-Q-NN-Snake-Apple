from snakeenv import SnakeEnv
from qdn import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

def train_dqn(env, agent, num_episodes, max_steps):
    rewards_per_episode = []
    steps_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        if len(agent.memory) > agent.batch_size:
            agent.replay()

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        print(f"Episode: {episode + 1}/{num_episodes}, Steps: {steps}, Reward: {total_reward}")

    return rewards_per_episode, steps_per_episode


env = SnakeEnv()

#Define the list of num_episodes
num_episodes_list = [15, 50, 100,200, 250] #10000

# Train the agent for each number of episodes
for num_episodes in num_episodes_list:
    # Create the DQNagent
    agent = DQNAgent(state_size=35, action_size=4, learning_rate=0.001,
                    gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=100)
    
    #Choose number of episodes and steps
    max_steps = 50
    
    #Training agent
    rewards, steps = train_dqn(env, agent, num_episodes, max_steps)
    
    # Plot rewards and steps per episode
    plt.plot(rewards)
    plt.title(f"Rewards per Episode (num_episodes={num_episodes})")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    
    plt.plot(steps)
    plt.title(f"Steps per Episode (num_episodes={num_episodes})")
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.show()
    
    #Save trained model
    model_filename = f"model_episodes_{num_episodes}.h5"
    agent.model.save_weights(model_filename)
    print(f"Trained model saved as {model_filename}")





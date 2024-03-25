from snakeenv import SnakeEnv
from qdn import DQNAgent
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

#Prompt model filename
model_filename = input("Enter the filename of the trained model (e.g., model_episodes_200.h5): ")

# Determine the number of episodes based on the model filename
if "model_episodes_200.h5" in model_filename:
    num_episodes = 200
elif "model_episodes_15.h5" in model_filename:
    num_episodes = 15
elif "model_episodes_50.h5" in model_filename:
    num_episodes = 50
elif "model_episodes_100.h5" in model_filename:
    num_episodes = 100
elif "model_episodes_250.h5" in model_filename:
    num_episodes = 250
else:
    print("Invalid model filename. Exiting the assessment.")
    exit()

#Load
agent = DQNAgent(state_size=35, action_size=4)
agent.model.load_weights(model_filename)


env = SnakeEnv()
max_steps = env.max_steps  # Maximum number of steps per episode

#Evaluate the performance (Effectiveness metric)
effectiveness = []

for _ in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.act(state)
        next_state, _, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        state = next_state
        steps += 1

    effectiveness.append(steps / max_steps)

#Accuracy and MSE
ground_truth_labels = np.zeros(num_episodes)
predictions = np.array(effectiveness) * max_steps
accuracy = accuracy_score(ground_truth_labels, predictions)
mse = mean_squared_error(ground_truth_labels, predictions)

#Print the evaluation results
print("Effectiveness:", np.mean(effectiveness))
print("Accuracy:", accuracy)
print("MSE:", mse)



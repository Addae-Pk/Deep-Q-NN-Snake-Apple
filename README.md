# Deep-Reinforcement-Learning-QDN-Snake-Apple
MLSMM2154 : Project part 2.2  [Snake Game with Deep Q-Network(DQN)]                      

In this part of the project, I designed a  Snake and apple game that has 
an AI agent that aims to learn and solve the Snake and Apple game using the Deep
Q-Network (DQN). This RL algorithm was choosed to deepen our use and
implementation of the Q-learning method.




## Project Files

1. `play_game.py`: This script allows you to play the Snake Game using a 
trained DRL agent. It loads a trained model and runs the game loop, where 
the agent makes decisions based on its learned policy. When runned, you'll 
be prompted to enter a filename (“model_episodes_15.h5”, 
“model_episodes_50.h5”, “model_episodes_100.h5”, or 
“model_episodes_200.h5”)

2. `qdn.py`: Deep Reinforcement Learning Agent that  learns to play the 
game by interacting with the environment and optimizing its actions based 
on rewards. The DQNAgent learns to play the Snake Game by training a 
neural network model using Q-learning and experience replay.

3. `snakeenv.py`:  defines the custom environment (SnakeEnv class), which 
implements the Snake Game environment. It provides methods for 
initializing the game, taking actions, updating the game state, and 
calculating rewards.

4. `train.py`: This script is used to train the DRL agent to play the 
Snake Game. It defines the training loop, where the agent interacts with 
the environment, collects experiences, and updates its neural network 
model based on the Q-learning algorithm.

5. `Assessment.py`: allows you to evaluate the performance of the trained model on the Snake game environment. It measures the accuracy and mean squared error (MSE) of the model's predictions compared to the ground truth scores.

	To use the assessment file, follow these steps:

		1. Run assessment.py
		2. Interact with the terminal and selected the model file  you are willing to use (for e.g`model_episodes_200.h5`). This will update the necessary evaluation parameters for this training size.





## Getting Started
To play the Snake Game using the trained agent:

1. Make sure you have Python 3.x installed on your system.

2. Install the required dependencies by running the following command:

   ```shell
   pip install -r requirements.txt

3. 
a)Run train.py (runtime -/+ 45min) and  run play_game.py (and choose 
the model with X number of episodes/training version you want the agent to 
play on by interacting in the terminal)  b) Directly run  the play_game.py and choose a trained model version 
(15, 50, 100 or 200 training episodes) by interaction with the terminal

        
## Dependencies : 
        - Python 3.9 or higher
        - Pygame
        - Numpy  
        - TensorFlow (or any other compatible deep learning library)

        Install the required dependencies: check requirement.txt file to
        install dependencies

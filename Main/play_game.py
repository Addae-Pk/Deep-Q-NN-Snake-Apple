from snakeenv import SnakeEnv
from qdn import DQNAgent
import numpy as np
import pygame


# Prompt for the filename of the trained model
model_filename = input("Enter the filename of the trained model (e.g., model_episodes_15.h5): ")

# Load the trained model
agent = DQNAgent(state_size=5 + 30, action_size=4)
agent.model.load_weights(model_filename)

# Create the environment
env = SnakeEnv()

# Run the game loop
running = True
while running:
    state = env.reset()
    state = np.reshape(state, [1, 5 + 30])

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Snake Game")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break

        if not running:
            break

        action = agent.choose_action(state)
        next_state, _, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 5 + 30])
        state = next_state

        screen.fill((0, 0, 0))
        for segment in env.snake_position:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(segment[0], segment[1], 10, 10))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(env.apple_position[0], env.apple_position[1], 10, 10))
        pygame.display.flip()
        clock.tick(5)




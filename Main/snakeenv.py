import gym
from gym import spaces
import numpy as np
import random
import pygame


class SnakeEnv(gym.Env):
    def __init__(self, snake_len_goal=30, max_steps=500):
        super(SnakeEnv, self).__init__()
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-500, high=500, shape=(5 + snake_len_goal,), dtype=np.float32)
        self.score = 0
        self.snake_len_goal = snake_len_goal
        self.button_direction = 1
        self.reward_apple = 50
        self.reward_collision = -5
        self.reward_step = -0.1
        self.max_steps = max_steps

    def step(self, action):
        self.button_direction = action
        self.move_snake()
        reward = self.calculate_reward()
        done = self.check_game_over()
        state = self.get_state()
        return state, reward, done, {}

    def reset(self):
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.button_direction = 1
        return self.get_state()

    def render(self):
        screen_width = 500
        screen_height = 500
        block_size = 10
    
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Snake Game")
    
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    
            self.move_snake()
    
            screen.fill((0, 0, 0))
    
            for segment in self.snake_position:
                pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(segment[0], segment[1], block_size, block_size))
    
            pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(self.apple_position[0], self.apple_position[1],
                                                              block_size, block_size))
    
            pygame.display.flip()
            clock.tick(10)
    
            state, reward, done, _ = self.step(self.button_direction)  # Add this line
    
            if done:
                running = False
    
        pygame.quit()


    
    def move_snake(self):
        if len(self.snake_position) > 0:
            if self.button_direction == 1:
                new_head = [self.snake_position[0][0] + 10, self.snake_position[0][1]]
            elif self.button_direction == 0:
                new_head = [self.snake_position[0][0] - 10, self.snake_position[0][1]]
            elif self.button_direction == 2:
                new_head = [self.snake_position[0][0], self.snake_position[0][1] + 10]
            elif self.button_direction == 3:
                new_head = [self.snake_position[0][0], self.snake_position[0][1] - 10]
    
            self.snake_position.insert(0, new_head)
    
            if self.snake_position[0] != self.apple_position:
                self.snake_position.pop()
            else:
                self.apple_position, self.score = self.collision_with_apple(self.apple_position, self.score)
                if len(self.snake_position) == self.snake_len_goal:
                    self.snake_len_goal += 10
                    self.snake_position.append([])
    
        return self.snake_position


    def collision_with_apple(self, apple_position, score):
        score += self.reward_apple
        return apple_position, score

    def calculate_reward(self):
        reward = self.reward_step

        if len(self.snake_position) > 0:
            if self.collision_with_boundaries(self.snake_position[0]) or self.collision_with_self(
                    self.snake_position):
                reward = self.reward_collision
            elif self.snake_position[0] == self.apple_position:
                reward = self.reward_apple

        return reward

    def check_game_over(self):
        if len(self.snake_position) > 0:
            return self.collision_with_boundaries(self.snake_position[0]) or self.collision_with_self(self.snake_position)
        else:
            return False

    def collision_with_boundaries(self, snake_head):
        if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
            return True
        else:
            return False

    def collision_with_self(self, snake_position):
        snake_head = snake_position[0]
        if snake_head in snake_position[1:]:
            return True
        else:
            return False

    def get_state(self):
        state = []

        if len(self.snake_position) > 0:
            state.extend(self.snake_position[0])
            state.extend(self.apple_position)

            snake_body = np.array(self.snake_position[1:])
            snake_body = snake_body.flatten()

            if len(snake_body) < self.snake_len_goal * 2:
                pad_length = (self.snake_len_goal * 2) - len(snake_body)
                snake_body = np.pad(snake_body, (0, pad_length), mode='constant')

            state.extend(snake_body)
            state = state[:35]

        if len(state) < self.observation_space.shape[0]:
            pad_length = self.observation_space.shape[0] - len(state)
            state = np.pad(state, (0, pad_length), mode='constant')

        return np.array(state)




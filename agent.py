import torch
import random
import numpy as np
from snake_game import QLearningSnakeGame, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from plotter import plot

MAXIMUM_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    
    def __init__(self):
        self.game_number = 0
        self.random_epsilon = 0
        self.discount_rate_gamma = 0.8
        self.memory = deque(maxlen=MAXIMUM_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.discount_rate_gamma)

    def get_state(self, game):

        head = game.snake[0]
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            #Left
            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_down and game.is_collision(point_right)),

            #Straight
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),

            #Right
            (direction_right and game.is_collision(point_down)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)),

            direction_left,
            direction_right,
            direction_up,
            direction_down,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def memorize(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            lesser_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            lesser_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*lesser_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        #exploration
        self.random_epsilon = 80 - self.game_number
        next_move = [0, 0, 0]
        if random.randint(0, 200) < self.random_epsilon:
            move = random.randint(0, 2)
            next_move[move] = 1
        else:
        #exploitation
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            next_move[move]=1
        
        return next_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    highscore = 0
    agent = Agent()
    environment = QLearningSnakeGame()

    while True:
        state_previous = agent.get_state(environment)

        previous_move = agent.get_action(state_previous)

        reward, game_over, score = environment.play_step(previous_move)

        next_state = agent.get_state(environment)

        agent.train_short_memory(state_previous, previous_move, reward, next_state, game_over)

        agent.memorize(state_previous, previous_move, reward, next_state, game_over)

        if game_over:
            environment.restart()
            agent.game_number += 1
            agent.train_long_memory()

            if score > highscore:
                highscore = score
                agent.model.save()

            print("Game iteration: ", agent.game_number, ", Score: ", score, ", Highscore: ", highscore) 

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.game_number
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

train()
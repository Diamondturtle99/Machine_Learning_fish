import numpy as np
import pygame
import sys
import time

# Game environment
class Game:
    def __init__(self):
        self.state = [(0, 0), (9, 9)]  # Starting positions of the two agents
        self.target = (np.random.randint(10), np.random.randint(10))  # Random target between (0, 0) and (9, 9)
        self.obstacles = self.generate_obstacles()  # Generate initial obstacles

    def get_state(self):
        return self.state

    def generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < 6:
            obstacle_x, obstacle_y = np.random.randint(10), np.random.randint(10)
            if (obstacle_x, obstacle_y) not in [self.state[0], self.state[1], self.target]:
                obstacles.add((obstacle_x, obstacle_y))
        return obstacles

    def take_action(self, actions):
        rewards = []

        for i, action in enumerate(actions):
            dx, dy = action
            x, y = self.state[i]
            x += dx
            y += dy
            x = max(0, min(9, x))  # Restrict x within the grid bounds
            y = max(0, min(9, y))  # Restrict y within the grid bounds

            if (x, y) in self.obstacles:
                # Agent hits an obstacle, revert the position back to the previous state
                x, y = self.state[i]

            self.state[i] = (x, y)

            if self.state[i] == self.target:
                reward = 1
                self.target = (np.random.randint(10), np.random.randint(10))  # Generate new target
                self.obstacles = self.generate_obstacles()  # Regenerate obstacles

            else:
                reward = -0.1

            rewards.append(reward)

        return rewards


# Q-Learning agent
class QLearningAgent:
    def __init__(self):
        self.Q = np.zeros((10, 10, 4))  # Q-table with 4 possible actions (up, down, left, right)
        self.learning_rate = 0.2  # Adjusted learning rate
        self.discount_factor = 0.99
        self.epsilon = 0.2  # Increased epsilon
        self.state = None

        # Initialize blue agent's Q-table with higher initial values
        self.initialize_blue_agent()

    def initialize_blue_agent(self):
        # Assign higher initial values to the blue agent's Q-table
        self.epsilon = 1.0
        self.Q[:, :, 0] = 1.0  # Up action
        self.Q[:, :, 1] = -1.0  # Down action
        self.Q[:, :, 2] = -1.0  # Left action
        self.Q[:, :, 3] = -1.0  # Right action

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random action
            action = np.random.randint(4)
        else:
            # Exploitation: Choose the action with the highest Q-value for the current state
            action = np.argmax(self.Q[state[0]])
        return action

    def update_q_table(self, state, action, reward, next_state):
        x, y = state[0]
        next_x, next_y = next_state[0]
        self.Q[x][y][action] = (1 - self.learning_rate) * self.Q[x][y][action] + self.learning_rate * (
                reward + self.discount_factor * np.max(self.Q[next_x][next_y]))

    def save_q_table(self, filename):
        np.save(filename, self.Q)

    def load_q_table(self, filename):
        self.Q = np.load(filename)


def draw_game_state(state, target, obstacles):
    # Draw the game state
    game_display.fill(white)

    # Draw the target
    target_x, target_y = target
    pygame.draw.rect(game_display, red, (target_x * cell_width, target_y * cell_height, cell_width, cell_height))

    # Draw the obstacles
    for obstacle in obstacles:
        obstacle_x, obstacle_y = obstacle
        pygame.draw.rect(game_display, black, (obstacle_x * cell_width, obstacle_y * cell_height, cell_width, cell_height))

    # Draw the agents
    pygame.draw.circle(game_display, blue,
                       (state[0][0] * cell_width + cell_width // 2, state[0][1] * cell_height + cell_height // 2),
                       min(cell_width, cell_height) // 3)
    pygame.draw.circle(game_display, green,
                       (state[1][0] * cell_width + cell_width // 2, state[1][1] * cell_height + cell_height // 2),
                       min(cell_width, cell_height) // 3)

    pygame.display.update()


def main():
    game = Game()
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()

    num_episodes = 5000  # Increased number of episodes
    save_interval = 1  # Save Q-table every second

    start_time = time.time()
    prev_time = start_time

    for episode in range(num_episodes):
        state = game.get_state()
        agent1.state = state
        agent2.state = state

        while True:
            action1 = agent1.get_action(state)
            action2 = agent2.get_action(state)

            actions = [[(0, -1), (0, 1), (-1, 0), (1, 0)][action1], [(0, -1), (0, 1), (-1, 0), (1, 0)][action2]]

            rewards = game.take_action(actions)

            next_state = game.get_state()

            agent1.update_q_table(state, action1, rewards[0], next_state)
            agent2.update_q_table(state, action2, rewards[1], next_state)

            state = next_state
            agent1.state = state
            agent2.state = state

            draw_game_state(game.state, game.target, game.obstacles)

            time.sleep(0.1)

            current_time = time.time()
            elapsed_time = current_time - prev_time

            # Save Q-table every second
            if elapsed_time >= save_interval:
                agent1.save_q_table('agent1_qtable.npy')
                agent2.save_q_table('agent2_qtable.npy')
                prev_time = current_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            if game.state[0] == game.target and game.state[1] == game.target:
                break

    # Save final Q-tables
    agent1.save_q_table('agent1_qtable.npy')
    agent2.save_q_table('agent2_qtable.npy')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    pygame.init()
    display_width, display_height = 800, 800
    game_display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('AI-Fishies!')

    cell_width = display_width // 10
    cell_height = display_height // 10

    blue = (0, 0, 255)
    green = (0, 255, 0)
    red = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)

    main()


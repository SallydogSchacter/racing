import argparse
import game_env
import pygame
import numpy as np
from dqn import DQNAgent
import time  # Import the time module

# Command-line arguments
parser = argparse.ArgumentParser(description="Racing DQN Training")
parser.add_argument("--load_weights", action="store_true", help="Load previous weights")
args = parser.parse_args()

TOTAL_GAMETIME = 1000  # Max game time for one episode
N_EPISODES = 10000
REPLACE_TARGET = 50
PENALTY = 100
NUM_CARS = 4  # Number of cars in the environment

game = game_env.RacingEnv(num_cars=NUM_CARS)
game.fps = 120

GameTime = 0
GameHistory = []
renderFlag = False
dqn_agent = DQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995, replace_target=REPLACE_TARGET, batch_size=1024, input_dims=19)

# Load weights if the argument is passed
if args.load_weights:
    try:
        dqn_agent.load_model("model_weights")
        print("Successfully loaded previous weights.")
    except FileNotFoundError:
        print("No previous weights found. Starting with random initialization.")


ddqn_scores = []
eps_history = []

def run():
    for e in range(N_EPISODES):
        
        # Dictionary to track step times
        step_times = {
            "Game Reset": 0,
            "Initial Observation/Reward": 0,
            "Action Selection": 0,
            "Environment Step": 0,
            "Memory Update": 0,
            "Learning": 0,
        }

        game_reset_time = time.time()
        game.reset()
        step_times["Game Reset"] = time.time() - game_reset_time

        # Initialize car states
        active_cars = [True] * NUM_CARS  # Track which cars are still active
        scores = [0] * NUM_CARS  # Scores for all cars
        counter = [0] * NUM_CARS  # Step counters for all cars

        observations, rewards, dones = game.step([0] * NUM_CARS)  # Initial actions for all cars
        prev_observations = [np.array(obs) if active_cars[i] else None for i, obs in enumerate(observations)]

        gtime = 0  # Set game time back to 0

        while any(active_cars):  # Run until all cars are done
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action_time = time.time()
            actions = [
                dqn_agent.choose_action(prev_observations[i]) if active_cars[i] else 0 for i in range(NUM_CARS)
            ]
            
            step_times["Action Selection"] += time.time() - action_time

            step_time = time.time()
            observations, rewards, dones = game.step(actions)
            observations = [
                np.array(observations[i]) if active_cars[i] else None for i in range(NUM_CARS)
            ]
            step_times["Environment Step"] = time.time() - step_time

            # Update scores and counters for active cars
            for i in range(NUM_CARS):
                if not active_cars[i]:
                    continue  # Skip cars that are no longer active
                if dones[i]:  # If the car is done, it crashed
                    active_cars[i] = False  # Mark car as inactive
                elif rewards[i] == 0:
                    counter[i] += 1
                    if counter[i] > 100:  # If car is idle for too long
                        scores[i] -= PENALTY
                        active_cars[i] = False  # Mark car as inactive
                else:
                    counter[i] = 0
                scores[i] += rewards[i]

                remember_time = time.time()
                dqn_agent.remember(
                    prev_observations[i], actions[i], rewards[i], observations[i], int(dones[i])
                )
                step_times["Memory Update"] += time.time() - remember_time

                learn_time = time.time()
                dqn_agent.learn()
                step_times["Learning"] += time.time() - learn_time

            # Update the game time
            gtime += 1

            # End the episode if the max time is reached
            if gtime >= TOTAL_GAMETIME:
                break

            # Render the game
            game.render(actions)

        # Track epsilon and main car's score
        eps_history.append(dqn_agent.epsilon)
        ddqn_scores.append(scores[0])
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])
        
        if e % 10 == 0 and e > 10:
            dqn_agent.save_model("model_weights")

        # Find the step with the maximum time
        max_step = max(step_times, key=step_times.get)
        max_time = step_times[max_step]
        
        print('episode: ', e,'score: %.2f' % scores[0],
              ' average score %.2f' % avg_score,
              ' epsilon: ', dqn_agent.epsilon,
              ' memory size', dqn_agent.memory.mem_cntr % dqn_agent.memory.mem_size)

        # print(
        #     f"Episode {e}: Max step time - {max_step} ({max_time:.4f} seconds)"
        # )

run()

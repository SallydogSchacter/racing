import game_env
import pygame
import numpy as np
from dqn import DQNAgent
import time  # Import the time module

TOTAL_GAMETIME = 1000  # Max game time for one episode
N_EPISODES = 10000
REPLACE_TARGET = 50
PENALTY = 100
NUM_CARS = 1  # Number of cars in the environment

game = game_env.RacingEnv(num_cars=NUM_CARS)
game.fps = 120

GameTime = 0
GameHistory = []
renderFlag = False
dqn_agent = DQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995, replace_target=REPLACE_TARGET, batch_size=1024, input_dims=19)

ddqn_scores = []
eps_history = []

def run():
    for e in range(N_EPISODES):
        game.reset()

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

            keys = pygame.key.get_pressed()
            actions = [0]
            # Map keys to actions
            if keys[pygame.K_w]:
                actions = [1]
            elif keys[pygame.K_a]:
                actions = [2]
            elif keys[pygame.K_s]:
                actions = [4]
            elif keys[pygame.K_d]:
                actions = [3]
    
            observations, rewards, dones = game.step(actions)
            observations = [
                np.array(observations[i]) if active_cars[i] else None for i in range(NUM_CARS)
            ]

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
                dqn_agent.learn()

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

        
        print('episode: ', e,'score: %.2f' % scores[0],
              ' average score %.2f' % avg_score,
              ' epsilon: ', dqn_agent.epsilon,
              ' memory size', dqn_agent.memory.mem_cntr % dqn_agent.memory.mem_size)

        # print(
        #     f"Episode {e}: Max step time - {max_step} ({max_time:.4f} seconds)"
        # )

run()


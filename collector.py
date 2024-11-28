import game_env
import pygame
from dqn import DQNAgent
import numpy as np

TOTAL_GAMETIME = 1000  # Max game time for one episode
N_EPISODES = 10000
REPLACE_TARGET = 50

game = game_env.RacingEnv()
game.fps = 30

GameTime = 0
GameHistory = []
renderFlag = False

dqn_agent = DQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995,
                     replace_target=REPLACE_TARGET, batch_size=512, input_dims=19)

ddqn_scores = []
eps_history = []


def run():
    output_file = "observations.txt"
    file = open(output_file, "a")
    for e in range(N_EPISODES):
        game.reset()  # reset env

        done = False
        score = 0
        counter = 0

        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        gtime = 0  # set game time back to 0

        renderFlag = True  # if you want to render every episode set to true
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            keys = pygame.key.get_pressed()
            action = 0
            # Map keys to actions
            if keys[pygame.K_w]:
                action = 4
            elif keys[pygame.K_a]:
                action = 2
            elif keys[pygame.K_s]:
                action = 1
            elif keys[pygame.K_d]:
                action = 3
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward
            file.write(f"{observation}, {action}, {reward}, {observation_}, {int(done)}\n")
            #dqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            #dqn_agent.learn()

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        eps_history.append(dqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])

    file.close()


run()

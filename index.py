import numpy as np
import random
import pygame
import sys


GRID_SIZE = 10


class SurvivalGridEnv:
    def __init__(self, size, num_zombies, num_supplies, num_obstacles):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
        # Inicializar
        self.zombies = self.place_items(num_zombies)
        self.supplies = self.place_items(num_supplies, exclude=self.zombies)
        self.obstacles = self.place_items(num_obstacles, exclude=self.zombies + self.supplies)

        self.collected_supplies = set()

        for pos in self.zombies:
            self.grid[pos] = 1
        for pos in self.supplies:
            self.grid[pos] = 2
        for pos in self.obstacles:
            self.grid[pos] = 3

    def place_items(self, num_items, exclude=[]):
        items = []
        while len(items) < num_items:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if pos not in items and pos not in exclude and pos != self.start and pos != self.goal:
                items.append(pos)
        return items

    def reset(self):
        self.agent_pos = self.start
        self.collected_supplies = set()
        return self.agent_pos, tuple(self.collected_supplies)
    
    # movimentos
    def step(self, action):
        i, j = self.agent_pos
        if action == 0: i = max(i-1, 0) # cima
        elif action == 1: i = min(i+1, self.size-1) # baixo
        elif action == 2: j = max(j-1, 0) # esquerda
        elif action == 3: j = min(j+1, self.size-1) # direitas

        if (i, j) in self.obstacles:
            i, j = self.agent_pos
        
        self.agent_pos = (i, j)
        
        reward, done = -0.1, False
        if self.agent_pos == self.goal:
            reward, done = 30, True  # Recompensa para alcançar a saída
        elif self.agent_pos in self.zombies:
            reward, done = -5, True  # Penalidade por encontrar um zumbi
        elif self.agent_pos in self.supplies and self.agent_pos not in self.collected_supplies:
            self.collected_supplies.add(self.agent_pos)
            reward = 10  # Recompensa para coletar um suprimento
        
        return self.agent_pos, tuple(self.collected_supplies), reward, done

    def render(self, screen, cell_size=60):
        colors = {
            "background": (200, 200, 200),
            "agent": (0, 0, 255),
            "goal": (0, 255, 0),
            "zombie": (0, 0, 0),
            "supply": (255, 215, 0),
            "obstacle": (100, 100, 100),
            "empty": (200, 200, 200)
            #0 150 0
        }

        screen.fill(colors["background"])
        for i in range(self.size):
            for j in range(self.size):
                color = colors["empty"]
                if (i, j) == self.agent_pos:
                    color = colors["agent"]
                elif (i, j) == self.goal:
                    color = colors["goal"]
                elif self.grid[i][j] == 1:
                    color = colors["zombie"]
                elif self.grid[i][j] == 2:
                    if (i, j) not in self.collected_supplies:
                        color = colors["supply"]
                elif self.grid[i][j] == 3:
                    color = colors["obstacle"]
                pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))

        pygame.display.flip()

# Configurações de aprendizagem do personagem
class QLearningAgent:
    def __init__(self, env, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.001):
        self.env = env
        self.q_table = np.zeros((env.size, env.size, 2 ** len(env.supplies), 4))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    # Exploração
    def policy(self, state, collected_supplies):
        index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in self.env.supplies]), 2)
        
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state[0], state[1], index])

    # Treina y atualiza
    def train(self, episodes, max_steps):
        final_supplies_collected = 0
        final_episode_result = ""
        final_score = 0

        for ep in range(episodes):
            state, collected_supplies = self.env.reset()
            episode_supplies_collected = 0
            episode_score = 0
            official_score = 0
            done = False
            
            for step in range(max_steps):
                action = self.policy(state, collected_supplies)
                next_state, next_collected_supplies, reward, done = self.env.step(action)

                index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in self.env.supplies]), 2)
                next_index = int(''.join(['1' if (i, j) in next_collected_supplies else '0' for (i, j) in self.env.supplies]), 2)

                best_next_action = np.max(self.q_table[next_state[0], next_state[1], next_index])
                self.q_table[state[0], state[1], index, action] += self.lr * (reward + self.gamma * best_next_action - self.q_table[state[0], state[1], index, action])

                state, collected_supplies = next_state, next_collected_supplies

                if reward == 10: 
                    episode_supplies_collected += 1

                episode_score += reward
                official_score += reward if reward != -0.1 else 0

                if done:
                    if reward == 30:
                        final_episode_result = f'Chegou ao destino no episódio {ep + 1}! Suprimentos coletados: {episode_supplies_collected}/{len(self.env.supplies)}'
                    elif reward == -5:
                        final_episode_result = f'Morreu ao encontrar um zumbi no episódio {ep + 1}. Suprimentos coletados: {episode_supplies_collected}/{len(self.env.supplies)}'
                    final_supplies_collected = episode_supplies_collected
                    final_score = int(official_score + reward) 
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))

            if ep % 1000 == 0:
                print(f'Episode {ep}')
                self.env.render(screen, cell_size)
                pygame.time.wait(500)

        # Cálculo da pontuação baseada nos presentes, saída e zumbis
        actual_final_score = final_supplies_collected * 10 + (30 if 'Chegou ao destino' in final_episode_result else 0) - (5 if 'Morreu ao encontrar um zumbi' in final_episode_result else 0)
        max_possible_score = len(self.env.supplies) * 10 + 30
        print(final_episode_result)
        print(f'Pontuação final: {actual_final_score}/{max_possible_score} pontos')


    def test(self):
        state, collected_supplies = self.env.reset()
        done = False
        episode_score = 0
        official_score = 0
        while not done:
            index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in self.env.supplies]), 2)
            action = np.argmax(self.q_table[state[0], state[1], index])
            next_state, next_collected_supplies, reward, done = self.env.step(action)
            self.env.render(screen, cell_size)
            pygame.time.wait(500)
            state, collected_supplies = next_state, next_collected_supplies
            episode_score += reward
            official_score += reward if reward != -0.1 else 0

        
        official_score = int(official_score + (reward if reward == -5 else 0)) 
        max_possible_score = len(self.env.supplies) * 10 + 30
        actual_final_score = len(collected_supplies) * 10 + (30 if reward == 30 else 0) - (5 if reward == -5 else 0)
        if reward == 30:
            print(f'Chegou ao destino! Suprimentos coletados: {len(collected_supplies)}/{len(self.env.supplies)}')
        elif reward == -5:
            print(f'Morreu ao encontrar um zumbi. Suprimentos coletados: {len(collected_supplies)}/{len(self.env.supplies)}')
        print(f'Pontuação final: {actual_final_score}/{max_possible_score} pontos')

        
        print("\nTabela de Pontuação:")
        print("Item          | Pontos")
        print("----------------------")
        print("Presente      | 10")
        print("Saída         | 30")
        print("Zumbi         | -5")

# Itens mapa
num_zombies = 10
num_supplies = 12  
num_obstacles = 8 

env = SurvivalGridEnv(GRID_SIZE, num_zombies, num_supplies, num_obstacles)
agent = QLearningAgent(env)

pygame.init()
cell_size = 60
screen = pygame.display.set_mode((env.size * cell_size, env.size * cell_size))
pygame.display.set_caption('Survival Grid')


agent.train(episodes=10000, max_steps=100)


agent.test()


pygame.quit()
sys.exit()

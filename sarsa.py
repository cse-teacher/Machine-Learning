import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

# --- Configuration ---
size = 5
goal = (0, 4)
obstacles = [(1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (3, 3)]
alpha = 0.2   # Learning rate (increased for faster demo)
gamma = 0.9   # Discount
epsilon = 0.1 # Exploration

class SARSAViz:
    def __init__(self):
        self.Q = np.zeros((size, size, 4)) # Q[row, col, action]
        self.state = (size-1, 0) # Start bottom-left
        self.action = self.choose_action(self.state)
        self.episodes = 0
        self.symbols = ['^', 'v', '<', '>'] # Up:0, Down:1, Left:2, Right:3
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 7))
        plt.subplots_adjust(bottom=0.2, top=0.85)
        
        # UI Elements
        ax_step = plt.axes([0.4, 0.05, 0.1, 0.06])
        self.btn_step = Button(ax_step, 'Move Rat', color='#99ff99')
        self.btn_step.on_clicked(self.step_move)

        ax_reset = plt.axes([0.52, 0.05, 0.1, 0.06])
        self.btn_reset = Button(ax_reset, 'Reset Rat', color='#ff9999')
        self.btn_reset.on_clicked(self.reset_rat)

        self.update_plot()
        plt.show()

    def choose_action(self, s):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)
        return np.argmax(self.Q[s[0], s[1], :])

    def step_move(self, event):
        row, col = self.state
        # Actions: 0:^, 1:v, 2:<, 3:>
        if self.action == 0: ns = (max(0, row - 1), col)
        elif self.action == 1: ns = (min(size - 1, row + 1), col)
        elif self.action == 2: ns = (row, max(0, col - 1))
        elif self.action == 3: ns = (row, min(size - 1, col + 1))
        
        if ns in obstacles: ns = self.state
        reward = 0 if ns == goal else -1
        
        next_action = self.choose_action(ns)
        
        # SARSA Update
        current_q = self.Q[self.state[0], self.state[1], self.action]
        next_q = self.Q[ns[0], ns[1], next_action]
        self.Q[self.state[0], self.state[1], self.action] += alpha * (reward + gamma * next_q - current_q)
        
        self.state = ns
        self.action = next_action
        
        if self.state == goal:
            self.episodes += 1
            self.reset_rat(None)
            
        self.update_plot()

    def reset_rat(self, event):
        self.state = (size-1, 0)
        self.action = self.choose_action(self.state)
        self.update_plot()

    def update_plot(self):
        for ax in self.axes: ax.clear()
        
        # PANEL 1: Environment & Rat
        self.axes[0].set_title(f"1. Environment (Ep: {self.episodes})")
        maze = np.zeros((size, size))
        for obs in obstacles: maze[obs] = -1
        self.axes[0].imshow(maze, cmap='binary', alpha=0.3)
        self.axes[0].text(goal[1], goal[0], 'CHEESE', ha='center', color='green', weight='bold')
        self.axes[0].text(self.state[1], self.state[0], 'RAT', ha='center', color='red', weight='bold', fontsize=12)

        # PANEL 2: Policy (Using Symbols ^, v, <, >)
        self.axes[1].set_title("2. Policy (Action Symbols)")
        self.axes[1].imshow(maze, cmap='binary', alpha=0.1)
        for r in range(size):
            for c in range(size):
                if (r, c) == goal or (r, c) in obstacles: continue
                # Only show symbol if the rat has actually learned something (Q != 0)
                if np.any(self.Q[r, c, :]):
                    best_a = np.argmax(self.Q[r, c, :])
                    symbol = self.symbols[best_a]
                    self.axes[1].text(c, r, symbol, ha='center', va='center', fontsize=20, color='blue', fontweight='bold')

        # PANEL 3: Value (Max Q)
        self.axes[2].set_title("3. Max Q-Value")
        v_map = np.max(self.Q, axis=2)
        im = self.axes[2].imshow(v_map, cmap='YlOrRd')
        for (r, c), val in np.ndenumerate(v_map):
            if (r, c) not in obstacles:
                self.axes[2].text(c, r, f'{val:.1f}', ha='center', va='center', color='black')
        
        plt.suptitle("SARSA Learning Demo", fontsize=16)
        self.fig.canvas.draw_idle()

viz = SARSAViz()
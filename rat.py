import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

# --- Configuration ---
size      = 5
goal      = (0, 4)
obstacles = [(1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (3, 3)]
gamma     = 0.9
actions   = ['^' ,'v','<' , '>' ]
class RLVisualizer:
    def __init__(self):
        self.V = np.zeros((size, size))
        self.policy = np.zeros((size, size), dtype=int)
        self.mode = "Value Iteration"
        
        # Setup Figure with 3 sections
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 7))
        plt.subplots_adjust(bottom=0.25) # Make room for UI
        
        # UI Elements
        ax_radio = plt.axes([0.15, 0.05, 0.2, 0.1], facecolor='#f0f0f0')
        self.radio = RadioButtons(ax_radio, ('Value Iteration', 'Policy Iteration'))
        self.radio.on_clicked(self.set_mode)

        ax_reset = plt.axes([0.45, 0.05, 0.1, 0.075])
        self.btn_reset = Button(ax_reset, 'Reset', color='#ff9999')
        self.btn_reset.on_clicked(self.reset)

        ax_step = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.btn_step = Button(ax_step, 'Step', color='#99ff99')
        self.btn_step.on_clicked(self.step)

        self.update_plot()
        plt.show()

    def set_mode(self, label):
        self.mode = label
        print(f"Switched to: {label}")

    def reset(self, event):
        self.V = np.zeros((size, size))
        self.policy = np.zeros((size, size), dtype=int)
        self.update_plot()

    def step(self, event):
        if self.mode == "Value Iteration":
            self.value_iteration_step()
        else:
            self.policy_iteration_step()
        self.update_plot()

    def value_iteration_step(self):
        new_V = np.copy(self.V)
        for r in range(size):
            for c in range(size):
                if (r, c) == goal or (r, c) in obstacles: continue
                v_list = []
                for a in range(4):
                    ns, rew = self.get_next_state_reward((r, c), a)
                    v_list.append(rew + gamma * self.V[ns])
                new_V[r, c] = max(v_list)
        self.V = new_V

    def policy_iteration_step(self):
        # 1. Simple Policy Evaluation (one sweep)
        new_V = np.copy(self.V)
        for r in range(size):
            for c in range(size):
                if (r, c) == goal or (r, c) in obstacles: continue
                a = self.get_best_action((r, c))
                ns, rew = self.get_next_state_reward((r, c), a)
                new_V[r, c] = rew + gamma * self.V[ns]
        self.V = new_V
        
    def get_next_state_reward(self, s, a):
        row, col = s
        if   a == 0: ns = (max(0, row - 1), col)
        elif a == 1: ns = (min(size - 1, row + 1), col)
        elif a == 2: ns = (row, max(0, col - 1))
        elif a == 3: ns = (row, min(size - 1, col + 1))
        if ns in obstacles: ns = s
        return ns, -1

    def get_best_action(self, s):
        v_list = []
        for a in range(4):
            ns, rew = self.get_next_state_reward(s, a)
            v_list.append(rew + gamma * self.V[ns])
        return np.argmax(v_list)

    def update_plot(self):
        for ax in self.axes: ax.clear()
        
        # 1. Environment
        self.axes[0].set_title("Environment")
        maze = np.zeros((size, size))
        for obs in obstacles: maze[obs] = -1
        self.axes[0].imshow(maze, cmap='binary')
        self.axes[0].text(goal[1], goal[0], 'GOAL', ha='center', color='green', weight='bold')

        # 2. Policy
        self.axes[1].set_title("Best Action")
        self.axes[1].imshow(self.policy, cmap='YlGn')
        self.axes[1].text(goal[1], goal[0], 'GOAL', ha='center', color='green', weight='bold')
        for (r, c), val in np.ndenumerate(self.policy):
                if (r, c) == goal or (r, c) in obstacles: continue
                a = self.get_best_action((r, c))
                self.axes[1].text(c, r, actions[a], ha='center')

        # 3. Value
        self.axes[2].set_title("Value Function")
        self.axes[2].imshow(self.V, cmap='YlGn')
        for (r, c), val in np.ndenumerate(self.V):
            if (r, c) not in obstacles:
                self.axes[2].text(c, r, f'{val:.1f}', ha='center')
        
        self.fig.canvas.draw_idle()

# Run the app
viz = RLVisualizer()
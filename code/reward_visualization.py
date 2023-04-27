import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class RewardVisualization():

    def __init__(self, root, max_episode, game, model_type):
        self.root = root
        self.max_episode = max_episode
        self.game = game
        self.reward_history = np.load(
            f"reward_history/{self.game}_{model_type}_{max_episode}.npy")
        self.model_type = model_type

    def show(self):

        def rolling_average(a, n):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        def update_plot(ax, window_size=1):
            ax.clear()
            ax.set_xlabel("Training episodes")
            ax.set_ylabel("Total episode reward")
            ax.set_title(f"Training rewards ({window_size} episode average)")
            ax.plot(np.arange(window_size,
                              len(self.reward_history) + 1),
                    rolling_average(self.reward_history, window_size))
            fig.canvas.draw_idle()

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        update_plot(ax)

        game = " ".join(self.game.split("_")).title()

        toplevel = tk.Toplevel(self.root)
        toplevel.title(
            f"{game} {self.model_type.upper()} Training Rewards Visualization")

        canvas = FigureCanvasTkAgg(fig, master=toplevel)
        canvas.get_tk_widget().pack()

        slider = tk.Scale(toplevel,
                          from_=1,
                          to=self.max_episode // 2,
                          resolution=1,
                          length=500,
                          orient="horizontal",
                          label="Window Size",
                          command=lambda x: update_plot(ax, int(x)))
        slider.pack()

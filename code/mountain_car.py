import time
import tkinter as tk

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

import utils
from reward_visualization import RewardVisualization


class MountainCar():

    def __init__(self, root):
        self.root = root
        self.game = "mountain_car"
        self.model_nums = {
            "a2c": utils.get_model_nums_int(f"{self.game}_a2c"),
            "dqn": utils.get_model_nums_int(f"{self.game}_dqn")
        }

    def get_models(self, model_type):
        models = {}
        for episodes in self.model_nums[model_type]:
            models[episodes] = tf.keras.models.load_model(
                f"models/{self.game}_{model_type}/{self.game}_{model_type}_{episodes}.h5"
            )
        return models

    def show_gameplay(self, episodes, model_type):
        closed = []
        utils.detect_close(closed)

        model = tf.keras.models.load_model(
            f"models/{self.game}_{model_type}/{self.game}_{model_type}_{episodes}.h5"
        )

        if not model:
            return

        env = gym.make("MountainCar-v0",
                       render_mode="human",
                       new_step_api=False)

        state = env.reset()

        while not closed:
            env.render()

            if model_type == "dqn":
                action = np.argmax(model(np.expand_dims(state, axis=0)))
            else:
                action_logits = model(np.expand_dims(state, axis=0))
                action = tf.random.categorical(action_logits, 1)[0, 0]

            next_state, _, done, _ = env.step(int(action))
            state = next_state

            if done:
                break

        time.sleep(2)
        env.close()

    def show_visualization(self, model_type):
        reward_visualization = RewardVisualization(
            self.root, max(self.model_nums[model_type]), self.game, model_type)
        reward_visualization.show()
        self.show_weight_visualization(self.get_models(model_type),
                                       self.model_nums[model_type], model_type)

    def show_weight_visualization(self, models, model_nums, model_type):

        def draw_histogram(model, ax):
            ax.clear()
            ax.set_xlabel("Weight")
            ax.set_ylabel("Frequency")
            ax.set_title("Weight distribution (64 bins)")
            ax.legend(handles=[
                Line2D([0], [0], color="#1f77b4", lw=4, label="Weights"),
                Line2D([0], [0], color="#ff7f0e", lw=4, label="Biases")
            ])
            for weight in model.layers[1].get_weights():
                ax.hist(np.ndarray.flatten(weight), bins=64)

        def draw_lines(model, ax):
            ax.clear()
            ax.set_xlabel("Connected nodes")
            ax.set_ylabel("Weight")
            ax.set_title("Node connections")
            ax.plot(model.layers[2].get_weights()[0])

        def update_plot(ax, value=25):
            model = models[value]
            draw_histogram(model, ax[0])
            draw_lines(model, ax[1])
            fig.canvas.draw_idle()

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        update_plot(ax)

        toplevel = tk.Toplevel(self.root)
        toplevel.title(
            f"Mountain Car {model_type.upper()} Model Weight Visualization")

        canvas = FigureCanvasTkAgg(fig, master=toplevel)
        canvas.get_tk_widget().pack()

        slider = tk.Scale(toplevel,
                          from_=model_nums[0],
                          to=model_nums[-1],
                          resolution=model_nums[1] - model_nums[0],
                          length=500,
                          orient="horizontal",
                          label="Training Episodes",
                          command=lambda x: update_plot(ax, int(x)))
        slider.pack()

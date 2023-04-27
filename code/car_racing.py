import time
import tkinter as tk
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import utils
from reward_visualization import RewardVisualization


class CarRacing():

    def __init__(self, root):
        self.root = root
        self.game = "car_racing"
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

        env = gym.make("CarRacing-v2",
                       render_mode="human",
                       continuous=False,
                       new_step_api=False)

        state = utils.CarRacing.process_state(env.reset())

        FRAME_STACK_SIZE = model.layers[0].input_shape[0][3]

        frame_stack = deque([state] * FRAME_STACK_SIZE, maxlen=FRAME_STACK_SIZE)

        while not closed:
            env.render()

            if model_type == "dqn":
                action = np.argmax(
                    model(
                        np.expand_dims(
                            utils.CarRacing.transpose_frame_stack(frame_stack),
                            axis=0)))
            else:
                action_logits = model(
                    np.expand_dims(
                        utils.CarRacing.transpose_frame_stack(frame_stack),
                        axis=0))
                action = tf.random.categorical(action_logits, 1)[0, 0]

            next_state, _, done, _ = env.step(int(action))
            frame_stack.append(utils.CarRacing.process_state(next_state))

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

        def get_conv_layer(model):
            for i, layer in enumerate(model.layers):
                if "conv2d" in layer.name:
                    return i

        def update_plot(fig, value=25):
            model = models[value]
            filters, _ = model.layers[self.conv_layer].get_weights()

            x, y = filters.shape[3], filters.shape[2]
            ax = fig.subplots(x, y)

            for i in range(x):
                for j in range(y):
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                    ax[i, j].imshow(filters[:, :, j, i])

            fig.canvas.draw_idle()

        self.conv_layer = get_conv_layer(models[model_nums[0]])

        fig, ax = plt.subplots(1, 1, figsize=(5, 8))
        fig.suptitle("Layer 1 filters")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Kernel")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        update_plot(fig)

        toplevel = tk.Toplevel(self.root)
        toplevel.title(
            f"Car Racing {model_type.upper()} Convolution Kernel Visualization")

        canvas = FigureCanvasTkAgg(fig, master=toplevel)
        canvas.get_tk_widget().pack()

        slider = tk.Scale(toplevel,
                          from_=model_nums[0],
                          to=model_nums[-1],
                          resolution=model_nums[1] - model_nums[0],
                          length=500,
                          orient="horizontal",
                          label="Training Episodes",
                          command=lambda x: update_plot(fig, int(x)))
        slider.pack()

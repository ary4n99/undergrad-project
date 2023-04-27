import datetime
import os
import threading
import time

import cv2
import numpy as np
import win32api


# GUI functions
def _detect(closed):
    while True:
        state = win32api.GetKeyState(0x51)
        if state != 0 and state != 1:
            closed.append(True)
        time.sleep(0.01)


def detect_close(closed):
    thread = threading.Thread(target=lambda: _detect(closed))
    thread.start()


# Common functions
def _process_name(s):
    l = s.split("_")
    return l[-1][:-3]


def get_model_nums_str(game):
    return [str(f) for f in get_model_nums_int(game)]


def get_model_nums_int(game):
    return sorted([int(_process_name(f)) for f in os.listdir(f"models/{game}")])


# Notebook functions
def _remove_file(path, game, episode):
    for file in os.listdir(path):
        if f"{game}_{episode}" in file:
            os.remove(os.path.join(path, file))


def save_progress(model, reward_history, episode, save_frequency, game):
    if not episode % save_frequency:
        model.save(f"models/{game}/{game}_{episode}.h5")

        np.save(f"reward_history/{game}_{episode}", np.array(reward_history))

        try:
            _remove_file("reward_history", game, episode - save_frequency)
        except Exception:
            pass


def log(episode, episode_reward, epsilon=None):
    print(
        f"{datetime.datetime.now()} - episode {episode} / reward: {episode_reward:.3f} {f'/ epsilon: {epsilon:.3f}' if epsilon else ''}"
    )


class CarRacing:

    @staticmethod
    def process_state(state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY).astype(float)
        cv2.rectangle(state, (45, 66), (50, 76), 0, -1)
        state[np.where((state >= 160) & (state < 180))] = 255
        state[np.where((state >= 90) & (state < 160))] = 100
        state = state[0:84, 6:90]
        state /= 255.0
        return state

    @staticmethod
    def transpose_frame_stack(frame_stack):
        return np.transpose(np.array(frame_stack), (1, 2, 0))

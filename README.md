# Deep Reinforcement Learning for Racing Games

![Python version](https://img.shields.io/badge/python-v3.7-blue)
!["Repository size"](https://img.shields.io/github/repo-size/ary4n99/undergrad-project)

Advantage Actor-Critic (A2C) and Deep Q-Network (DQN) implementations for MountainCar-v0 and CarRacing-v2 using TensorFlow 2, accompanied by an undergraduate thesis.

## Getting Started

> **NOTE: pywin32 is used for the GUI.**
>
> **To run on a non-Windows machine, remove lines 8-22 (inclusive) from `code/utils.py`; this will remove window closing functionality for the gameplay visualization.**

1. Clone the repository

    ```
    git clone https://github.com/ary4n99/undergrad-project.git
    cd undergrad-project
    ```

2. Install `python3.7`

    *Recommended: create a new conda environment*

    ```
    conda create -n env_name python=3.7
    conda activate env_name
    ```

3. Install the required dependencies

    ```
    cd code
    pip install -r requirements.txt
    ```

4. Run the GUI

    ```
    python gui.py
    ```

## Results

|   |  Mountain Car | Car Racing |
|---|---|---|
| Advantage Actor-Critic (A2C) | ![mountain_car_a2c](assets/recordings/mountain_car_a2c_475.gif) | ![car_racing_a2c](assets/recordings/car_racing_a2c_475.gif) |
| Deep Q-Network (DQN) | ![mountain_car_dqn](assets/recordings/mountain_car_dqn_500.gif) | ![car_racing_dqn](assets/recordings/car_racing_dqn_500.gif) |

## Training Rewards

|  Mountain Car | Car Racing |
|---|---|
| ![mountain_car](assets/training_rewards/mountain_car.png)| ![car_racing](assets/training_rewards/car_racing.png) |

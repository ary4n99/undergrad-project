import tkinter as tk
from tkinter import ttk
import warnings

import utils
from car_racing import CarRacing
from mountain_car import MountainCar

warnings.filterwarnings("ignore", category=DeprecationWarning)


class GUI():

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Select Options")
        self.button_width = 30

        self.mode = tk.StringVar(self.root, value="gameplay")
        self.game = tk.StringVar(self.root, value="mountain_car")
        self.model_type = tk.StringVar(self.root, value="a2c")

        self.model_num = tk.StringVar(self.root)
        self.menu_items = {}

        self.mountain_car = MountainCar(self.root)
        self.car_racing = CarRacing(self.root)

        self.mountain_car_a2c_model_nums = utils.get_model_nums_str(
            "mountain_car_a2c")
        self.car_racing_a2c_model_nums = utils.get_model_nums_str(
            "car_racing_a2c")
        self.mountain_car_dqn_model_nums = utils.get_model_nums_str(
            "mountain_car_dqn")
        self.car_racing_dqn_model_nums = utils.get_model_nums_str(
            "car_racing_dqn")

        self.create_style()
        self.create_menu_items()
        self.create_menu()

        self.root.mainloop()

    def create_style(self):
        style = ttk.Style(self.root)

        style.configure("IndicatorOff.TRadiobutton",
                        indicatorrelief=tk.FLAT,
                        indicatormargin=-1,
                        indicatordiameter=-1,
                        relief=tk.RAISED,
                        focusthickness=0,
                        highlightthickness=0,
                        padding=10)

        style.map("IndicatorOff.TRadiobutton",
                  background=[("selected", "white"), ("active", "#ececec")])

    def create_menu_items(self):
        self.menu_items = {
            "gameplay":
                ttk.Radiobutton(self.root,
                                text="Gameplay",
                                variable=self.mode,
                                value="gameplay",
                                width=self.button_width,
                                command=self.toggle_select_model,
                                style="IndicatorOff.TRadiobutton"),
            "visualization":
                ttk.Radiobutton(self.root,
                                text="Model Visualization",
                                variable=self.mode,
                                value="visualization",
                                width=self.button_width,
                                command=self.toggle_select_model,
                                style="IndicatorOff.TRadiobutton"),
            "mountain_car":
                ttk.Radiobutton(self.root,
                                text=f"Mountain Car (1D)",
                                variable=self.game,
                                value="mountain_car",
                                width=self.button_width,
                                command=self.toggle_select_model,
                                style="IndicatorOff.TRadiobutton"),
            "car_racing":
                ttk.Radiobutton(self.root,
                                text=f"Car Racing (2D)",
                                variable=self.game,
                                value="car_racing",
                                width=self.button_width,
                                command=self.toggle_select_model,
                                style="IndicatorOff.TRadiobutton"),
            "a2c":
                ttk.Radiobutton(self.root,
                                text=f"Advantage Actor Critic (A2C)",
                                variable=self.model_type,
                                value="a2c",
                                width=self.button_width,
                                command=self.toggle_select_model,
                                style="IndicatorOff.TRadiobutton"),
            "dqn":
                ttk.Radiobutton(self.root,
                                text=f"Deep Q Network (DQN)",
                                variable=self.model_type,
                                value="dqn",
                                width=self.button_width,
                                command=self.toggle_select_model,
                                style="IndicatorOff.TRadiobutton"),
            "run_button":
                ttk.Button(self.root, text="Run", width=20, command=self.run),
            "exit_label":
                ttk.Label(self.root,
                          text="Press Q to exit gameplay",
                          font="Helvetica 8 italic")
        }

    def create_menu(self):
        self.menu_items["gameplay"].grid(row=0, column=0)
        self.menu_items["visualization"].grid(row=0, column=1)
        self.menu_items["mountain_car"].grid(row=1, column=0)
        self.menu_items["car_racing"].grid(row=1, column=1)
        self.menu_items["a2c"].grid(row=2, column=0)
        self.menu_items["dqn"].grid(row=2, column=1)
        self.toggle_select_model()
        self.menu_items["run_button"].grid(row=5,
                                           columnspan=2,
                                           padx=10,
                                           pady=10)

    def toggle_select_model(self):
        try:
            self.menu_items["select_model_button"].grid_remove()
            self.menu_items["episodes_label"].grid_remove()
            self.menu_items["exit_label"].grid_remove()
        except Exception:
            pass

        self.model_num.set("25")
        mode = self.mode.get()
        game = self.game.get()
        model = self.model_type.get()
        model_nums = []

        if game == "mountain_car" and model == "a2c":
            model_nums = self.mountain_car_a2c_model_nums
        if game == "mountain_car" and model == "dqn":
            model_nums = self.mountain_car_dqn_model_nums
        if game == "car_racing" and model == "a2c":
            model_nums = self.car_racing_a2c_model_nums
        if game == "car_racing" and model == "dqn":
            model_nums = self.car_racing_dqn_model_nums

        if mode == "gameplay":
            self.menu_items["select_model_button"] = ttk.Combobox(
                self.root,
                values=model_nums,
                textvariable=self.model_num,
                state="readonly")
            self.menu_items["select_model_button"].grid(
                row=4,
                columnspan=2,
                padx=10,
            )

            self.menu_items["episodes_label"] = ttk.Label(
                self.root, text="Training episodes")
            self.menu_items["episodes_label"].grid(row=3,
                                                   columnspan=2,
                                                   padx=10,
                                                   pady=(10, 5))
            self.menu_items["exit_label"].grid(row=6,
                                               columnspan=2,
                                               padx=10,
                                               pady=(0, 10))

    def run(self):
        mode = self.mode.get()
        game = self.game.get()
        model = self.model_type.get()

        if mode == "gameplay" and game == "mountain_car":
            self.mountain_car.show_gameplay(int(self.model_num.get()), model)
        elif mode == "gameplay" and game == "car_racing":
            self.car_racing.show_gameplay(int(self.model_num.get()), model)
        elif mode == "visualization" and game == "mountain_car":
            self.mountain_car.show_visualization(model)
        elif mode == "visualization" and game == "car_racing":
            self.car_racing.show_visualization(model)


if __name__ == "__main__":
    GUI()

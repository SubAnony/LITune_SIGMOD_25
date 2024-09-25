import numpy as np
import random
import time
import os
from tqdm import tqdm
from math import exp
import time

def simulated_annealing(param_grid, initial_temperature, cooling_rate, max_iterations, model_loss, time_budget=9999):
    start_time = time.monotonic()
    current_params = {}
    for key, value in param_grid.items():
        current_params[key] = random.choice(value)

    params = (list(current_params.values()))

    current_loss = model_loss(params)
    best_params = params
    best_loss = current_loss
    temperature = initial_temperature

    for iteration in tqdm(range(max_iterations)):
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > time_budget:
            break

        new_params_dict = {}
        for key, value in param_grid.items():
            new_params_dict[key] = random.choice(value)
        new_params = (list(new_params_dict.values()))

        new_loss = model_loss(new_params)

        if new_loss < current_loss or exp((current_loss - new_loss) / temperature) > random.random():
            params = new_params
            current_loss = new_loss

        if new_loss < best_loss:
            best_params = new_params
            best_loss = new_loss

        temperature *= cooling_rate
        print("current best loss:", best_loss)

    return best_params, best_loss
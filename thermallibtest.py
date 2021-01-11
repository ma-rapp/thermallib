import numpy as np
import os
from thermallib.chip import Chip


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'testmodel')
SVG_FILENAME = os.path.join(HERE, 'testmodel-steady.svg')


def test():
    tdp = 100
    processorsY, processorsX = 8, 8
    inactive_power = 0.2
    config = {}
    chip = Chip.load_or_create(MODEL_PATH, processorsY, processorsX, tdp=tdp, ambient_temperature=45, max_temperature=80, inactive_power=inactive_power, config=config)
    powers = np.zeros((processorsY, processorsX)) + tdp / chip.cores
    assert abs(chip.get_steady_state(powers).max() - 80) < 0.1

    """
    active_cores = np.zeros((processorsY, processorsX))
    active_cores[2,2] = 1
    active_cores[5,2] = 1
    active_cores[2,5] = 1
    active_cores[5,5] = 1
    powers = active_cores * chip.tsp_for_mapping(active_cores) + (1 - active_cores) * inactive_power
    chip.plot_grid_steady_state(powers, SVG_FILENAME, pure=False)
    """


def test_power_budget_const_temperature_for_mapping():
    inactive_power = 0.3
    chip = Chip.load(MODEL_PATH, inactive_power=inactive_power)
    active_cores = np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    powers = chip.power_budget_max_steady_state(active_cores)
    steady_state = chip.get_steady_state(powers)
    for y in range(chip.processorsY):
        for x in range(chip.processorsX):
            if active_cores[y,x]:
                assert abs(steady_state[y,x] - 80) < 0.01


if __name__ == '__main__':
    test()
    test_power_budget_const_temperature_for_mapping()

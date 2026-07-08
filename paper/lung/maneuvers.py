import numpy as np

timestep = 0.5

def normal_cycle(p_low, p_high, t_low, t_high, t_ramp):
    return np.concatenate((
        np.linspace(p_low, p_high, int(t_ramp / timestep)+1),
        np.ones(int(t_high / timestep)-1) * p_high,
        np.linspace(p_high, p_low, int(t_ramp / timestep)+1),
        np.ones(int(t_low / timestep)-1) * p_low,
    ))

def repeat_cycle(cycle, num_repeat):
    return np.concatenate([cycle for _ in range(num_repeat)])


maneuver_1 = repeat_cycle(normal_cycle(800, 2000, 3.5, 2.5, 0.5), 6)
maneuver_2 = np.concatenate((
    np.linspace(800, 2500, int(21 / timestep)+1)[:-1],
    np.linspace(2500, 800, int(21 / timestep)+1)[:-1],
))
maneuver_3 = np.concatenate((
    normal_cycle(800, 2000, 3.5, 2.5, 0.5),
    normal_cycle(800, 1500, 3.5, 2.5, 0.5),
    normal_cycle(800, 1000, 3.5, 2.5, 0.5),
    normal_cycle(800, 2000, 3.5, 2.5, 0.5),
    normal_cycle(800, 1200, 3.5, 2.5, 0.5),
    normal_cycle(800, 1800, 3.5, 2.5, 0.5),
))
maneuver_4 = np.concatenate((
    normal_cycle(800, 2000, 11.5, 8.5, 0.5),
    normal_cycle(800, 1500, 11.5, 8.5, 0.5)
))
maneuvers = [maneuver_1, maneuver_2, maneuver_3, maneuver_4]
time = np.arange(0, len(maneuvers[0])) * 0.5

import pickle
with open("output/maneuvers.pickle", 'wb') as handle:
    pickle.dump({"time": time, "pressure": maneuvers}, handle, protocol=pickle.HIGHEST_PROTOCOL)
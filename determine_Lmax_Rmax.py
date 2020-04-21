import numpy as np
from scipy.optimize import fsolve

speed_of_light_in_fiber = 204190.477  # km / s
attenuation_length = 22  # km


def rate(elementary_link_length, number_of_repeaters, number_of_modes, swap_probability):
    """Assumes dual-rail encoding and BSM success probability 50% at the midpoint station."""

    one_mode_link_prob = .5 * np.exp(- elementary_link_length / attenuation_length)
    link_prob = 1 - np.power(1 - one_mode_link_prob, number_of_modes)
    success_prob = np.power(link_prob, number_of_repeaters + 1) * np.power(swap_probability, number_of_repeaters)
    return speed_of_light_in_fiber / elementary_link_length * success_prob


def fidelity(elementary_link_fidelity, number_of_repeaters):

    depolar_prob = (4 * elementary_link_fidelity - 1) / 3
    return .25 + .75 * np.power(depolar_prob, number_of_repeaters + 1)


def solve_number_of_repeaters(elementary_link_fidelity, target_fidelity):

    def f(number_of_repeaters):
        calculated_fidelity = fidelity(elementary_link_fidelity=elementary_link_fidelity, number_of_repeaters=number_of_repeaters)
        return calculated_fidelity - target_fidelity

    number_of_repeaters = fsolve(func=f, x0=1)
    # return number_of_repeaters
    return np.floor(number_of_repeaters)


def solve_rate(number_of_repeaters, number_of_modes, swap_probability, target_rate):

    def f(elementary_link_length):
        calculated_rate = rate(elementary_link_length=elementary_link_length,
                               number_of_repeaters=number_of_repeaters,
                               swap_probability=swap_probability,
                               number_of_modes=number_of_modes)
        return calculated_rate - target_rate

    elementary_link_length = fsolve(func=f, x0=50)
    return np.floor(elementary_link_length)


def max_length_and_rate(target_fidelity, target_rate, elementary_link_fidelity, number_of_modes, swap_probability):

    [Rmax] = solve_number_of_repeaters(elementary_link_fidelity=elementary_link_fidelity,
                                       target_fidelity=target_fidelity)
    [Lmax] = solve_rate(number_of_repeaters=Rmax,
                        target_rate=target_rate,
                        number_of_modes=number_of_modes,
                        swap_probability=swap_probability)

    print("Requirements\n\ntarget_fidelity: {}\ntarget_rate: {} Hz\n\n"
          "Parameters\n\nelementary_link_fidelity: {}\nnumber_of_modes: {}\nswap_probability: {}\n\n"
          "Results\n\nLmax = {} km\nRmax = {} \n\n"
          .format(target_fidelity, target_rate, elementary_link_fidelity, number_of_modes, swap_probability, Lmax, Rmax))

    return Lmax, Rmax


if __name__ == "__main__":

    max_length_and_rate(target_fidelity=0.93,
                        target_rate=1,
                        elementary_link_fidelity=0.99,
                        number_of_modes=1000,
                        swap_probability=.5)

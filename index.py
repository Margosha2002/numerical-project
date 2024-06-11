import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time


J = 1
kb = 1
L = 100
N = L**2


def init_state(initial_state):
    AA = []
    if initial_state == 1 or initial_state == -1:
        for i in range(L):
            line = []
            for j in range(L):
                line.append(initial_state)
            AA.append(line)
        return AA
    if initial_state == 0:
        for i in range(L):
            line = []
            for j in range(L):
                if random.random() <= 0.5:
                    line.append(1)
                else:
                    line.append(-1)
            AA.append(line)
        return AA
    else:
        raise ValueError("Invalid initial_state")


def hamiltonian(A):  # count hamiltonian of state A
    sum = 0
    for i in range(L):
        for j in range(L):
            sum += A[i][j] * (A[i - 1][j] + A[i][j - 1])
    return -1 * J * sum


def probability_plus(
    A, i, j, time
):  # return probability of changing current spin into +1
    k = A[i - 1][j] + A[i][j - 1]
    if i == L - 1:
        k += A[0][j]
    else:
        k += A[i + 1][j]
    if j == L - 1:
        k += A[i][0]
    else:
        k += A[i][j + 1]
    return 1 / (1 + math.exp(-2 * (1 / (kb * time)) * k))


def magnetization(A):
    return float(np.sum(A)) / N  # return magnetization


def energy(A):
    return hamiltonian(A) / N  # return energy


def mean(arr):  # return mean
    sum1 = 0
    for i in arr:
        sum1 += i
    return sum1 / len(arr)


def magnetic_susceptibility(m, t):  # betta*N*⟨m**2⟩ − ⟨m⟩**2
    m2 = m
    for i in range(len(m2)):
        m2[i] = m2[i] ** 2
    d = mean(m2) - mean(m) ** 2
    return N * d / (kb * t)


def specific_heat(e, t):  # betta**2*N*⟨m**2⟩ − ⟨m⟩**2
    e2 = []
    for i in range(len(e)):
        e2.append(e[i] ** 2)
    d = mean(e2) - mean(e) ** 2
    return N * d / ((kb * t) ** 2)


def simulate_states(init_st, t, a, b, step):
    states_logs = []
    A = init_state(init_st)
    for i in range(b * step + 1):
        if i % step == 0 and i >= a * step and i <= b * step:
            state = []
            for i in range(L):
                line = []
                for j in range(L):
                    line.append(A[i][j])
                state.append(line)
            states_logs.append(state)
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        if random.random() <= probability_plus(A, i, j, t):
            A[i][j] = 1
        else:
            A[i][j] = -1
    return states_logs


def simulation(init_st, t):  # return magnetization and energy of every N-th step
    magnetization_logs = []
    energy_logs = []
    A = init_state(init_st)
    tt = time.time()
    for i in range(1000000):
        if i % N == 0:
            magnetization_logs.append(magnetization(A))
            energy_logs.append(energy(A))
        # time.sleep(0.1)
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        if random.random() <= probability_plus(A, i, j, t):
            A[i][j] = 1
        else:
            A[i][j] = -1
    print("it takes " + str(time.time() - tt))
    return magnetization_logs, energy_logs


def note():  # draws graphics of magnetization and energy
    magnet11, ener11 = simulation(1, 2)
    magnet12, ener12 = simulation(0, 2)
    magnet13, ener13 = simulation(-1, 2)
    magnet21, ener21 = simulation(1, 2.5)
    magnet22, ener22 = simulation(0, 2.5)
    magnet23, ener23 = simulation(-1, 2.5)
    figure, axis = plt.subplots(2, 2)
    axis[0, 0].plot(magnet11)
    axis[0, 0].plot(magnet12)
    axis[0, 0].plot(magnet13)
    axis[0, 0].legend(
        [
            "initial state = 1",
            "initial state = 1 with probability 1/2",
            "initial state = -1",
        ]
    )
    axis[0, 0].set_title("magnetization, T = 2")
    axis[0, 1].plot(magnet21)
    axis[0, 1].plot(magnet22)
    axis[0, 1].plot(magnet23)
    axis[0, 1].legend(
        [
            "initial state = 1",
            "initial state = 1 with probability 1/2",
            "initial state = -1",
        ]
    )
    axis[0, 1].set_title("magnetization, T = 2.5")
    axis[1, 0].plot(ener11)
    axis[1, 0].plot(ener12)
    axis[1, 0].plot(ener13)
    axis[1, 0].legend(
        [
            "initial state = 1",
            "initial state = 1 with probability 1/2",
            "initial state = -1",
        ]
    )
    axis[1, 0].set_title("energy, T = 2")
    axis[1, 1].plot(ener21)
    axis[1, 1].plot(ener22)
    axis[1, 1].plot(ener23)
    axis[1, 1].legend(
        [
            "initial state = 1",
            "initial state = 1 with probability 1/2",
            "initial state = -1",
        ]
    )
    axis[1, 1].set_title("energy, T = 2.5")
    plt.show()


def note1():  # draws mean magnetization, mean energy, magnetic susceptibility and specific heat as functions of time
    t = []
    for i in range(1, 41):
        t.append(i * 0.1)
    mean_magnetization = []
    mean_energy = []
    magnetic_susceptibility_logs = []
    specific_heat_logs = []
    for tt in t:
        m, e = simulation(0, tt)
        mean_magnetization.append(mean(m))
        mean_energy.append(mean(e))
        magnetic_susceptibility_logs.append(magnetic_susceptibility(m, tt))
        specific_heat_logs.append(specific_heat(e, tt))
    figure, axis = plt.subplots(2, 2)
    axis[0, 0].plot(mean_magnetization, "o")
    axis[0, 0].set_title("mean magnetization, T [0, 4]")
    axis[0, 1].plot(mean_energy, "o")
    axis[0, 1].set_title("mean energy, T [0, 4]")
    axis[1, 0].plot(magnetic_susceptibility_logs, "o")
    axis[1, 0].set_title("magnetic susceptibility, T [0, 4]")
    axis[1, 1].plot(specific_heat_logs, "o")
    axis[1, 1].set_title("specific heat, T [0, 4]")
    plt.show()


def note2(t):  # draws microscopic configurations
    s1 = simulate_states(0, t, 1, 9, 2000)
    figure, axis = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axis[i, j].pcolor(s1[3 * i + j])
    plt.show()

# External (3rd party)
import numpy as np

# External (ours)
from openwfs.devices import SLM
from openwfs.simulation import SimulatedWFS


def phase_response_test_function(phi, b, c, gamma):
    """A synthetic phase response function: 2π*(b + c*(phi/2π)^gamma)"""
    return np.clip(2 * np.pi * (b + c * (phi / (2 * np.pi)) ** gamma), 0, None)


def inverse_phase_response_test_function(f, b, c, gamma):
    """Inverse of the synthetic phase response function: 2π*(b + c*(phi/2π)^gamma)"""
    return 2 * np.pi * ((f / (2 * np.pi) - b) / c) ** (1 / gamma)


def lookup_table_test_function(f, b, c, gamma):
    """
    Compute the lookup indices (i.e. a lookup table)
    for countering the synthetic phase response test function: 2π*(b + c*(phi/2π)^gamma).
    """
    phase = inverse_phase_response_test_function(f, b, c, gamma)
    return (np.mod(phase, 2 * np.pi) * 256 / (2 * np.pi) + 0.5).astype(np.uint8)


# Simulated phase response settings
b = 0.1
c = 0.6
gamma = 1.15
linear_phase = np.arange(0, 2 * np.pi, 2 * np.pi / 256)

size = (10, 10)
t = np.random.normal(size=size) + 1j * np.random.normal(size=size)
sim = SimulatedWFS(t=t)

sim.slm.phase_response = phase_response_test_function(linear_phase, b, c, gamma)
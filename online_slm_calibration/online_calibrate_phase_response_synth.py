# External 3rd party
import torch
import numpy as np
from calibration_functions import learn_field, predict_feedback
from online_slm_calibration.plot_utilities import plot_results_ground_truth

# === Settings === #
settings = {
    "do_plot": True,
    "plot_per_its": 30,
    "nonlinearity": 2,
    "learning_rate": 0.3,
    "iterations": 1800,
    "smooth_loss_factor": 0,
}

# construct ground truth
noise_level = 0.3

phase_gt = 4.0 * np.pi * torch.linspace(0.0, 1.0, 256) ** 2
a_gt = 5.0
b_gt = 20.0

gv0 = np.arange(0, 256)
gv1 = np.arange(0, 256, 32)

measurements = predict_feedback(gv0, gv1, a_gt, b_gt, phase_gt, nonlinearity=settings['nonlinearity'], noise_level=noise_level)

lr, nonlinearity, phase, amplitude = learn_field(gray_values0=gv0, gray_values1=gv1, measurements=measurements, **settings)

print(f'b = {amplitude.mean()} ({b_gt}), B = {lr} (1.0)')

plot_results_ground_truth(phase, amplitude, phase_gt)

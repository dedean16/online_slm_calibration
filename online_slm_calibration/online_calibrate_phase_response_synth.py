# External 3rd party
import torch
import numpy as np
import matplotlib.pyplot as plt
from calibration_functions import learn_field, predict_feedback

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


nonlinearity, lr, phase, amplitude = learn_field(gray_values0=gv0, gray_values1=gv1, measurements=measurements, **settings)

print(f'b = {amplitude.mean()} ({b_gt}), B = {lr} (1.0)')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(phase_gt, color='C0', label='Ground truth')
plt.plot(phase, '--', color='C1', label='Predicted')
plt.xlabel('Gray value')
plt.ylabel('Phase response')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(amplitude, color='C0', label='Amplitude')
plt.show()

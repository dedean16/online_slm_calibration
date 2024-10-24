# External 3rd party
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# External ours
from openwfs.algorithms.troubleshoot import field_correlation

# Internal
from calibration_functions import predict_feedback, grow_learn_lut, learn_field


# === Settings === #
do_plot = True
do_end_plot = False
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)

noise_level = 0.3


phase_response_per_gv_gt = 4.0 * np.pi * torch.linspace(0.0, 1.0, 256) ** 2
a_gt = torch.tensor(5.0)
b_gt = torch.tensor(20.0)

gv0 = torch.arange(0, 256, dtype=torch.int32)
gv1 = torch.arange(0, 256, 32, dtype=torch.int32)


feedback_meas = predict_feedback(gv0, gv1, a_gt, b_gt, phase_response_per_gv_gt, nonlinearity=N, noise_level=noise_level)


lr, phase_response_per_gv_fit, amplitude = learn_field(gray_values0=gv0,
                gray_values1=gv1,measurements=feedback_meas, nonlinearity=N,
                learning_rate=0.2, iterations=500, do_plot=do_plot, do_end_plot=do_end_plot,
                plot_per_its=3, smooth_loss_factor=2.0, phase_stroke_init=12 )

print(f'b = {amplitude.mean()} ({b_gt}), lr = {lr} (1.0)')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(phase_response_per_gv_gt, color='C0', label='Ground truth')
plt.plot(phase_response_per_gv_fit, '--', color='C1', label='Predicted')
plt.xlabel('Gray value')
plt.ylabel('Phase response')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(amplitude, color='C0', label='Amplitude')
plt.show()

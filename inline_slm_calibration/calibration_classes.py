# External (3rd party)
import numpy as np
from numpy import ndarray as nd

# External (ours)
from openwfs import Processor, Detector, PhaseSLM
from openwfs.utilities import project


class OnlineSLMCalibrator:
    """
    Online SLM Calibrator

    Calibrate an SLM phase response based on feedback measurements.
    """
    def __init__(self, feedback: Detector, slm: PhaseSLM, num_of_phase_steps=256):
        self.feedback = feedback
        self.slm = slm
        self.num_of_phase_steps = num_of_phase_steps
        self.phase_response = None

    def calibrate(self):
        Nx = 4
        Ny = 4
        static_phase = 0

        # Create checkerboard phase mask
        y_check = np.arange(Ny).reshape((Ny, 1))
        x_check = np.arange(Nx).reshape((1, Nx))
        checkerboard = np.mod(x_check + y_check, 2)
        static_phase_pattern = static_phase * (1 - checkerboard)

        # Random SLM pattern to destroy focus
        self.slm.set_phases(2 * np.pi * np.random.rand(300, 300))
        self.slm.update()

        # Read dark frame
        dark_frame = self.feedback.read()
        dark_var = dark_frame.var()

        count = 0
        data_shape = self.feedback.data_shape
        img_stack = np.zeros((data_shape[0], data_shape[1], self.num_of_phase_steps))
        phase_range = np.linspace(0, 2 * np.pi, self.num_of_phase_steps)
        for phase in phase_range:
            phase_pattern = static_phase_pattern + phase * checkerboard
            self.slm.set_phases(phase_pattern)
            self.slm.update()

            img_stack[:, :, count] = self.feedback.read().copy()
            count += 1

        self.slm.set_phases(2 * np.pi * np.random.rand(300, 300))
        self.slm.update()

        # Plot frames
        # fig = plt.figure()
        # for n in range(num_of_phase_steps):
        #     fig.clear()
        #     plt.imshow(img_stack[:, :, n].squeeze())
        #     plt.title(f'{n}')
        #     plt.draw()
        #     plt.pause(1e-3)

        # STD of frame, corrected for background noise
        stack_std_corrected = np.sqrt((img_stack.var(axis=(0, 1)) - dark_var).clip(min=0))
        block_size_pix = int(slm.shape[0] / Ny)
        return self.phase_response


class TwoPhotonMicroscope(Processor):
    """
    Simple simulated Two Photon Microscope, with stationary focus on specimen center. Returns the signal at
    the sample plane. In future versions, we might return a scanned image.
    """
    def __init__(self, specimen, aberrations, incident_field, aperture_mask, gaussian_noise_std_per_pixel=0.0):
        super().__init__(specimen, aberrations, incident_field, aperture_mask, multi_threaded=False)
        self.pupil_field = None
        self.gaussian_noise_std_per_pixel = gaussian_noise_std_per_pixel
        self._data_shape = ()

    def _fetch(self, specimen_data: nd, aberrations_data: nd, incident_field_data: nd, aperture_mask_data: nd) -> nd:
        self.pupil_field = incident_field_data * aberrations_data * aperture_mask_data
        pupil_field_proj = project(self.pupil_field, out_shape=specimen_data.shape, out_extent=(4, 4))
        self.sample_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupil_field_proj))) \
            / pupil_field_proj.size
        N = 2
        two_photon_fluorescence = np.abs(self.sample_field)**(2*N)
        signal = specimen_data * two_photon_fluorescence

        # Uncomment for debugging
        # plt.cla()
        # plot_field(np.concatenate((pupil_field_proj, 0.2*np.abs(sample_field)**2), axis=1), 1)
        # plt.pause(0.01)
        noise_sample = np.random.randn() * self.gaussian_noise_std_per_pixel * specimen_data.sum()
        return signal.sum() + noise_sample

    @property
    def data_shape(self):
        """Returns the shape of the image in the image plane"""
        return self._data_shape

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class NoiseGenerator:
    """
    Class to generate different types of noise
    """

    @staticmethod
    def gaussian_noise(points, std):
        return np.random.normal(0, std, len(points))

    @staticmethod
    def percentage_jitter(points, percentage):
        jitter = np.random.uniform(-percentage / 100, percentage / 100, len(points))
        return points * jitter

    @staticmethod
    def frequency_noise(angles, freq_hz, amplitude, current_time):
        """Generate noise based on frequency"""
        return amplitude * np.sin(2 * np.pi * freq_hz * current_time + angles)


def generate_noisy_circle(radius=1.0, num_points=360, center=(0, 0),
                          gaussian_std=0.1, sine_amplitude=0.1, sine_freq=4,
                          percent_jitter=5, freq_noise_hz=30, freq_amplitude=0.1):
    """
    Generate points forming a circle with multiple types of noise.

    Parameters:
    -----------
    radius : float
        Radius of the base circle
    num_points : int
        Number of points to generate
    center : tuple
        (x, y) coordinates of circle center
    gaussian_std : float
        Standard deviation for Gaussian noise
    sine_amplitude : float
        Amplitude of sinusoidal noise
    sine_freq : int
        Frequency of sinusoidal noise
    percent_jitter : float
        Percentage of point value to use for jitter
    freq_noise_hz : float
        Frequency in Hz for time-based noise
    freq_amplitude : float
        Amplitude of frequency-based noise
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Generate base circle
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Add Gaussian noise
    if gaussian_std > 0:
        x += NoiseGenerator.gaussian_noise(x, gaussian_std)
        y += NoiseGenerator.gaussian_noise(y, gaussian_std)

    # Add sinusoidal noise
    if sine_amplitude > 0:
        radial_noise = sine_amplitude * np.sin(sine_freq * angles)
        x += radial_noise * np.cos(angles)
        y += radial_noise * np.sin(angles)

    # Add percentage-based jitter
    if percent_jitter > 0:
        x += NoiseGenerator.percentage_jitter(x, percent_jitter)
        y += NoiseGenerator.percentage_jitter(y, percent_jitter)

    # Center offset
    x += center[0]
    y += center[1]

    return x, y, angles


class CircleSampler:
    def __init__(self, x, y, angles, period=2 * np.pi, freq_noise_hz=30, freq_amplitude=0.1):
        self.x = x
        self.y = y
        self.angles = angles
        self.period = period
        self.current_time = 0
        self.freq_noise_hz = freq_noise_hz
        self.freq_amplitude = freq_amplitude

        # Initialize visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.plot_circle()
        self.current_point = None

    def plot_circle(self):
        """Plot the base circle with all points"""
        self.ax.clear()
        self.ax.plot(self.x, self.y, 'b.', alpha=0.3, label='Circle Points')
        self.ax.grid(True)
        self.ax.set_title('Noisy Circle Sampling')
        self.ax.legend()

    def sample_next_point(self, time_gap):
        """Get the next point with frequency-based noise"""
        self.current_time += time_gap

        # Convert time to angle
        target_angle = (2 * np.pi * self.current_time / self.period) % (2 * np.pi)

        # Find closest point
        idx = np.argmin(np.abs(self.angles - target_angle))

        # Add frequency-based noise
        if self.freq_noise_hz > 0:
            freq_noise = NoiseGenerator.frequency_noise(
                self.angles[idx:idx + 1],
                self.freq_noise_hz,
                self.freq_amplitude,
                self.current_time
            )
            x = self.x[idx] + freq_noise[0] * np.cos(self.angles[idx])
            y = self.y[idx] + freq_noise[0] * np.sin(self.angles[idx])
        else:
            x, y = self.x[idx], self.y[idx]

        # Update plot
        self.update_plot(x, y)

        return x, y

    def update_plot(self, x, y):
        """Update the plot with the current sampled point"""
        if self.current_point:
            self.current_point.remove()
        self.current_point = self.ax.plot(x, y, 'ro', markersize=10,
                                          label='Current Point')[0]
        self.ax.legend()
        plt.pause(0.01)  # Small pause to update the plot


def demo_sampling():
    """Demonstrate the circle generation and sampling"""
    # Generate noisy circle
    x, y, angles = generate_noisy_circle(
        radius=1.0,
        num_points=360,
        gaussian_std=0.05,
        sine_amplitude=0.1,
        sine_freq=3,
        percent_jitter=5,
        freq_noise_hz=30,
        freq_amplitude=0.05
    )

    # Create sampler with visualization
    sampler = CircleSampler(x, y, angles, period=10.0,
                            freq_noise_hz=30, freq_amplitude=0.05)

    # Sample points every 0.1 seconds
    try:
        while True:
            point = sampler.sample_next_point(time_gap=0.1)
            print(f"Current point: {point}")
            time.sleep(0.1)  # Real-time delay
    except KeyboardInterrupt:
        plt.close()


if __name__ == "__main__":
    demo_sampling()
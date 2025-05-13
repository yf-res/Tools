import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time
from enum import Enum
from scipy.interpolate import interp1d


class SamplingStrategy(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    INTERPOLATED = "interpolated"


class NoiseGenerator:
    """Enhanced noise generator with visualization capabilities"""

    @staticmethod
    def gaussian_noise(points, std):
        noise = np.random.normal(0, std, len(points))
        return noise, f"Gaussian (std={std:.3f})"

    @staticmethod
    def percentage_jitter(points, percentage):
        jitter = np.random.uniform(-percentage / 100, percentage / 100, len(points))
        noise = points * jitter
        return noise, f"Jitter ({percentage}%)"

    @staticmethod
    def frequency_noise(angles, freq_hz, amplitude, current_time):
        noise = amplitude * np.sin(2 * np.pi * freq_hz * current_time + angles)
        return noise, f"Frequency ({freq_hz}Hz)"

    @staticmethod
    def combined_noise(points, angles, params, current_time=0):
        """Generate all noise components and return individually"""
        noises = []
        if params.get('gaussian_std', 0) > 0:
            noise, label = NoiseGenerator.gaussian_noise(points, params['gaussian_std'])
            noises.append((noise, label))

        if params.get('percent_jitter', 0) > 0:
            noise, label = NoiseGenerator.percentage_jitter(points, params['percent_jitter'])
            noises.append((noise, label))

        if params.get('freq_noise_hz', 0) > 0:
            noise, label = NoiseGenerator.frequency_noise(
                angles, params['freq_noise_hz'],
                params['freq_amplitude'], current_time
            )
            noises.append((noise, label))

        return noises


class CircleVisualizer:
    """Handle all visualization aspects"""

    def __init__(self, figsize=(15, 10)):
        self.fig = plt.figure(figsize=figsize)
        self.gs = GridSpec(2, 2, self.fig)

        # Main circle plot
        self.ax_circle = self.fig.add_subplot(self.gs[0, 0])
        self.ax_circle.set_aspect('equal')

        # Noise components plot
        self.ax_noise = self.fig.add_subplot(self.gs[0, 1])

        # Time series plot
        self.ax_time = self.fig.add_subplot(self.gs[1, :])

        # Initialize time series data
        self.time_data = []
        self.x_data = []
        self.y_data = []

        self.fig.tight_layout()

    def plot_circle(self, x, y, noise_components=None):
        """Plot main circle and noise components"""
        self.ax_circle.clear()
        self.ax_circle.plot(x, y, 'b.', alpha=0.3, label='Circle Points')
        self.ax_circle.grid(True)
        self.ax_circle.set_title('Noisy Circle')
        self.ax_circle.legend()

        if noise_components:
            self.ax_noise.clear()
            angles = np.linspace(0, 2 * np.pi, len(x))
            for noise, label in noise_components:
                self.ax_noise.plot(angles, noise, label=label, alpha=0.7)
            self.ax_noise.set_title('Noise Components')
            self.ax_noise.set_xlabel('Angle (radians)')
            self.ax_noise.set_ylabel('Amplitude')
            self.ax_noise.legend()
            self.ax_noise.grid(True)

    def update_current_point(self, x, y, time_val):
        """Update current point and time series"""
        # Update circle plot
        self.ax_circle.plot(x, y, 'ro', markersize=10)

        # Update time series
        self.time_data.append(time_val)
        self.x_data.append(x)
        self.y_data.append(y)

        # Keep only last 100 points for visualization
        if len(self.time_data) > 100:
            self.time_data = self.time_data[-100:]
            self.x_data = self.x_data[-100:]
            self.y_data = self.y_data[-100:]

        self.ax_time.clear()
        self.ax_time.plot(self.time_data, self.x_data, 'b-', label='X coordinate')
        self.ax_time.plot(self.time_data, self.y_data, 'r-', label='Y coordinate')
        self.ax_time.set_title('Position Time Series')
        self.ax_time.set_xlabel('Time (s)')
        self.ax_time.set_ylabel('Position')
        self.ax_time.legend()
        self.ax_time.grid(True)

        plt.pause(0.01)


class CircleSampler:
    def __init__(self, x, y, angles, period=2 * np.pi, noise_params=None,
                 strategy=SamplingStrategy.UNIFORM):
        self.x = x
        self.y = y
        self.angles = angles
        self.period = period
        self.current_time = 0
        self.noise_params = noise_params or {}
        self.strategy = strategy

        # Initialize interpolators for smooth sampling
        self.x_interp = interp1d(angles, x, kind='cubic', fill_value='extrapolate')
        self.y_interp = interp1d(angles, y, kind='cubic', fill_value='extrapolate')

        # Initialize visualizer
        self.visualizer = CircleVisualizer()
        self.plot_initial_state()

    def plot_initial_state(self):
        """Plot initial circle state with noise components"""
        noise_components = NoiseGenerator.combined_noise(
            self.x, self.angles, self.noise_params, self.current_time
        )
        self.visualizer.plot_circle(self.x, self.y, noise_components)

    def get_next_angle(self, time_gap):
        """Get next angle based on sampling strategy"""
        if self.strategy == SamplingStrategy.UNIFORM:
            return (2 * np.pi * self.current_time / self.period) % (2 * np.pi)

        elif self.strategy == SamplingStrategy.RANDOM:
            return np.random.uniform(0, 2 * np.pi)

        elif self.strategy == SamplingStrategy.ADAPTIVE:
            # Adaptive sampling based on noise level
            base_angle = (2 * np.pi * self.current_time / self.period) % (2 * np.pi)
            noise_level = sum(abs(n[0][0]) for n in NoiseGenerator.combined_noise(
                [1], [base_angle], self.noise_params, self.current_time
            ))
            # Adjust sampling rate based on noise level
            return base_angle + noise_level * np.random.uniform(-0.1, 0.1)

        elif self.strategy == SamplingStrategy.INTERPOLATED:
            base_angle = (2 * np.pi * self.current_time / self.period) % (2 * np.pi)
            return base_angle

    def sample_next_point(self, time_gap):
        """Sample next point using selected strategy"""
        self.current_time += time_gap
        target_angle = self.get_next_angle(time_gap)

        if self.strategy == SamplingStrategy.INTERPOLATED:
            # Use interpolation for smooth sampling
            x = float(self.x_interp(target_angle))
            y = float(self.y_interp(target_angle))
        else:
            # Find closest point for other strategies
            idx = np.argmin(np.abs(self.angles - target_angle))
            x, y = self.x[idx], self.y[idx]

        # Add frequency-based noise if specified
        if self.noise_params.get('freq_noise_hz', 0) > 0:
            noise, _ = NoiseGenerator.frequency_noise(
                np.array([target_angle]),
                self.noise_params['freq_noise_hz'],
                self.noise_params['freq_amplitude'],
                self.current_time
            )
            x += noise[0] * np.cos(target_angle)
            y += noise[0] * np.sin(target_angle)

        # Update visualization
        self.visualizer.update_current_point(x, y, self.current_time)

        return x, y


def demo_sampling(strategy=SamplingStrategy.UNIFORM):
    """Demonstrate circle sampling with specified strategy"""
    # Generate base circle with noise
    num_points = 360
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    noise_params = {
        'gaussian_std': 0.05,
        'percent_jitter': 5,
        'freq_noise_hz': 30,
        'freq_amplitude': 0.05
    }

    # Apply initial noise
    for noise, _ in NoiseGenerator.combined_noise(x, angles, noise_params):
        x += noise * np.cos(angles)
        y += noise * np.sin(angles)

    # Create sampler
    sampler = CircleSampler(x, y, angles, period=10.0,
                            noise_params=noise_params,
                            strategy=strategy)

    # Sample points
    try:
        while True:
            point = sampler.sample_next_point(time_gap=0.1)
            print(f"Current point: {point}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        plt.close()


if __name__ == "__main__":
    # Demo with different sampling strategies
    strategies = [
        SamplingStrategy.UNIFORM,
        SamplingStrategy.RANDOM,
        SamplingStrategy.ADAPTIVE,
        SamplingStrategy.INTERPOLATED
    ]

    for strategy in strategies:
        print(f"\nDemonstrating {strategy.value} sampling strategy...")
        demo_sampling(strategy)
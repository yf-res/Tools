import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.fft import fft, fftfreq
import time
from enum import Enum
from scipy.interpolate import interp1d


class SamplingStrategy(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    INTERPOLATED = "interpolated"


class CircleVisualizer:
    """Enhanced visualizer with phase plots, FFT analysis, and interactive controls"""

    def __init__(self, figsize=(15, 12)):
        self.fig = plt.figure(figsize=figsize)

        # Create grid for plots and controls
        self.gs = GridSpec(3, 3, height_ratios=[1, 1, 0.2])

        # Main circle plot
        self.ax_circle = self.fig.add_subplot(self.gs[0, 0])
        self.ax_circle.set_aspect('equal')

        # Noise components plot
        self.ax_noise = self.fig.add_subplot(self.gs[0, 1])

        # Phase plot
        self.ax_phase = self.fig.add_subplot(self.gs[0, 2])
        self.ax_phase.set_aspect('equal')

        # Time series plot
        self.ax_time = self.fig.add_subplot(self.gs[1, 0])

        # FFT plot
        self.ax_fft = self.fig.add_subplot(self.gs[1, 1])

        # Velocity plot
        self.ax_velocity = self.fig.add_subplot(self.gs[1, 2])

        # Initialize data storage
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.vx_data = []
        self.vy_data = []

        # Create sliders
        self.create_sliders()

        # Create strategy selector
        self.create_strategy_selector()

        self.fig.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for controls

    def create_sliders(self):
        """Create interactive sliders for parameter adjustment"""
        # Create axes for sliders
        slider_color = 'lightgoldenrodyellow'
        self.ax_gaussian = plt.axes([0.1, 0.1, 0.3, 0.03], facecolor=slider_color)
        self.ax_jitter = plt.axes([0.1, 0.06, 0.3, 0.03], facecolor=slider_color)
        self.ax_freq = plt.axes([0.1, 0.02, 0.3, 0.03], facecolor=slider_color)

        # Create sliders
        self.slider_gaussian = Slider(self.ax_gaussian, 'Gaussian σ', 0.0, 0.2, valinit=0.05)
        self.slider_jitter = Slider(self.ax_jitter, 'Jitter %', 0.0, 20.0, valinit=5.0)
        self.slider_freq = Slider(self.ax_freq, 'Freq (Hz)', 0.0, 50.0, valinit=30.0)

        # Store current values
        self.current_params = {
            'gaussian_std': 0.05,
            'percent_jitter': 5.0,
            'freq_noise_hz': 30.0,
            'freq_amplitude': 0.05
        }

    def create_strategy_selector(self):
        """Create radio buttons for sampling strategy selection"""
        self.ax_radio = plt.axes([0.5, 0.02, 0.15, 0.15], facecolor='lightgoldenrodyellow')
        self.radio = RadioButtons(self.ax_radio,
                                  [s.value for s in SamplingStrategy],
                                  active=0)

    def get_current_params(self):
        """Get current parameter values from sliders"""
        self.current_params['gaussian_std'] = self.slider_gaussian.val
        self.current_params['percent_jitter'] = self.slider_jitter.val
        self.current_params['freq_noise_hz'] = self.slider_freq.val
        return self.current_params

    def get_current_strategy(self):
        """Get current sampling strategy from radio buttons"""
        return SamplingStrategy(self.radio.value_selected)

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

    def update_analysis(self, x, y, time_val):
        """Update all analysis plots"""
        # Store data
        self.time_data.append(time_val)
        self.x_data.append(x)
        self.y_data.append(y)

        # Calculate velocities
        if len(self.time_data) > 1:
            dt = self.time_data[-1] - self.time_data[-2]
            vx = (self.x_data[-1] - self.x_data[-2]) / dt
            vy = (self.y_data[-1] - self.y_data[-2]) / dt
            self.vx_data.append(vx)
            self.vy_data.append(vy)

        # Keep only last 200 points
        max_points = 200
        if len(self.time_data) > max_points:
            self.time_data = self.time_data[-max_points:]
            self.x_data = self.x_data[-max_points:]
            self.y_data = self.y_data[-max_points:]
            self.vx_data = self.vx_data[-max_points:]
            self.vy_data = self.vy_data[-max_points:]

        # Update time series
        self.ax_time.clear()
        self.ax_time.plot(self.time_data, self.x_data, 'b-', label='X')
        self.ax_time.plot(self.time_data, self.y_data, 'r-', label='Y')
        self.ax_time.set_title('Position Time Series')
        self.ax_time.set_xlabel('Time (s)')
        self.ax_time.set_ylabel('Position')
        self.ax_time.legend()
        self.ax_time.grid(True)

        # Update phase plot
        self.ax_phase.clear()
        self.ax_phase.plot(self.x_data, self.y_data, 'g-', alpha=0.5)
        self.ax_phase.plot(x, y, 'ro')
        self.ax_phase.set_title('Phase Plot')
        self.ax_phase.set_xlabel('X Position')
        self.ax_phase.set_ylabel('Y Position')
        self.ax_phase.grid(True)

        # Update FFT plot
        if len(self.time_data) > 1:
            self.ax_fft.clear()
            dt = np.mean(np.diff(self.time_data))
            n = len(self.x_data)
            freq = fftfreq(n, dt)[:n // 2]

            fft_x = np.abs(fft(self.x_data))[:n // 2]
            fft_y = np.abs(fft(self.y_data))[:n // 2]

            self.ax_fft.plot(freq, fft_x, 'b-', label='X FFT', alpha=0.7)
            self.ax_fft.plot(freq, fft_y, 'r-', label='Y FFT', alpha=0.7)
            self.ax_fft.set_title('FFT Analysis')
            self.ax_fft.set_xlabel('Frequency (Hz)')
            self.ax_fft.set_ylabel('Magnitude')
            self.ax_fft.legend()
            self.ax_fft.grid(True)

        # Update velocity plot
        if len(self.vx_data) > 0:
            self.ax_velocity.clear()
            self.ax_velocity.plot(self.vx_data, self.vy_data, 'k.', alpha=0.5)
            self.ax_velocity.set_title('Velocity Space')
            self.ax_velocity.set_xlabel('X Velocity')
            self.ax_velocity.set_ylabel('Y Velocity')
            self.ax_velocity.grid(True)

        plt.pause(0.01)


class CircleSampler:
    """Enhanced sampler with interactive parameter updates"""

    def __init__(self, x, y, angles, period=2 * np.pi, noise_params=None,
                 strategy=SamplingStrategy.UNIFORM):
        self.x = x
        self.y = y
        self.angles = angles
        self.period = period
        self.current_time = 0
        self.noise_params = noise_params or {}
        self.strategy = strategy

        # Initialize interpolators
        self.x_interp = interp1d(angles, x, kind='cubic', fill_value='extrapolate')
        self.y_interp = interp1d(angles, y, kind='cubic', fill_value='extrapolate')

        # Initialize visualizer
        self.visualizer = CircleVisualizer()
        self.plot_initial_state()

        # Set up callbacks
        self.setup_callbacks()

    def setup_callbacks(self):
        """Set up callbacks for interactive controls"""
        self.visualizer.slider_gaussian.on_changed(self.update_params)
        self.visualizer.slider_jitter.on_changed(self.update_params)
        self.visualizer.slider_freq.on_changed(self.update_params)
        self.visualizer.radio.on_clicked(self.update_strategy)

    def update_params(self, val):
        """Update noise parameters from sliders"""
        self.noise_params = self.visualizer.get_current_params()
        self.plot_initial_state()

    def update_strategy(self, label):
        """Update sampling strategy from radio buttons"""
        self.strategy = SamplingStrategy(label)

    def plot_initial_state(self):
        """Plot initial circle state with noise components"""
        noise_components = NoiseGenerator.combined_noise(
            self.x, self.angles, self.noise_params, self.current_time
        )
        self.visualizer.plot_circle(self.x, self.y, noise_components)

    def sample_next_point(self, time_gap):
        """Sample next point with updated parameters"""
        self.current_time += time_gap
        self.strategy = self.visualizer.get_current_strategy()
        target_angle = self.get_next_angle(time_gap)

        if self.strategy == SamplingStrategy.INTERPOLATED:
            x = float(self.x_interp(target_angle))
            y = float(self.y_interp(target_angle))
        else:
            idx = np.argmin(np.abs(self.angles - target_angle))
            x, y = self.x[idx], self.y[idx]

        # Add dynamic noise
        if self.noise_params.get('freq_noise_hz', 0) > 0:
            noise, _ = NoiseGenerator.frequency_noise(
                np.array([target_angle]),
                self.noise_params['freq_noise_hz'],
                self.noise_params['freq_amplitude'],
                self.current_time
            )
            x += noise[0] * np.cos(target_angle)
            y += noise[0] * np.sin(target_angle)

        # Update visualization with new analysis
        self.visualizer.update_analysis(x, y, self.current_time)

        return x, y

    def get_next_angle(self, time_gap):
        """Get next angle based on current strategy"""
        if self.strategy == SamplingStrategy.UNIFORM:
            return (2 * np.pi * self.current_time / self.period) % (2 * np.pi)
        elif self.strategy == SamplingStrategy.RANDOM:
            return np.random.uniform(0, 2 * np.pi)
        elif self.strategy == SamplingStrategy.ADAPTIVE:
            base_angle = (2 * np.pi * self.current_time / self.period) % (2 * np.pi)
            noise_level = sum(abs(n[0][0]) for n in NoiseGenerator.combined_noise(
                [1], [base_angle], self.noise_params, self.current_time
            ))
            return base_angle + noise_level * np.random.uniform(-0.1, 0.1)
        else:  # INTERPOLATED
            return (2 * np.pi * self.current_time / self.period) % (2 * np.pi)


def demo_sampling():
    """Demonstrate interactive circle sampling"""
    # Generate base circle
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

    # Create sampler
    sampler = CircleSampler(x, y, angles, period=10.0,
                            noise_params=noise_params)

    # Sample points
    try:
        while True:
            point = sampler.sample_next_point(time_gap=0.1)
            time.sleep(0.1)
    except KeyboardInterrupt:
        plt.close()


if __name__ == "__main__":
    demo_sampling()
import numpy as np


def generate_noisy_circle(radius=1.0, num_points=360, center=(0, 0),
                          gaussian_std=0.1, sine_amplitude=0.1, sine_freq=4):
    """
    Generate points forming a circle with both Gaussian and sinusoidal noise.

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
        Frequency of sinusoidal noise (number of complete waves around circle)

    Returns:
    --------
    tuple
        (x_coordinates, y_coordinates, angles)
    """
    # Generate angles evenly spaced around the circle
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Generate base circle
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Add Gaussian noise
    if gaussian_std > 0:
        x += np.random.normal(0, gaussian_std, num_points)
        y += np.random.normal(0, gaussian_std, num_points)

    # Add sinusoidal noise
    if sine_amplitude > 0:
        radial_noise = sine_amplitude * np.sin(sine_freq * angles)
        x += radial_noise * np.cos(angles)
        y += radial_noise * np.sin(angles)

    # Add center offset
    x += center[0]
    y += center[1]

    return x, y, angles


class CircleSampler:
    """
    Class to sample points from a noisy circle based on time gaps.
    """

    def __init__(self, x, y, angles, period=2 * np.pi):
        """
        Initialize the sampler with circle points and period.

        Parameters:
        -----------
        x : array-like
            x coordinates of the circle points
        y : array-like
            y coordinates of the circle points
        angles : array-like
            angles corresponding to each point
        period : float
            Time period for one complete revolution
        """
        self.x = x
        self.y = y
        self.angles = angles
        self.period = period
        self.current_time = 0

    def sample_next_point(self, time_gap):
        """
        Get the next point on the circle based on a time gap.

        Parameters:
        -----------
        time_gap : float
            Time difference to the next point

        Returns:
        --------
        tuple
            (x, y) coordinates of the next point
        """
        self.current_time += time_gap

        # Convert time to angle based on the period
        target_angle = (2 * np.pi * self.current_time / self.period) % (2 * np.pi)

        # Find the closest angle in our sample points
        idx = np.argmin(np.abs(self.angles - target_angle))

        return self.x[idx], self.y[idx]


# Example usage
if __name__ == "__main__":
    # Generate noisy circle
    x, y, angles = generate_noisy_circle(
        radius=1.0,
        num_points=360,
        gaussian_std=0.05,
        sine_amplitude=0.1,
        sine_freq=3
    )

    # Create sampler
    sampler = CircleSampler(x, y, angles, period=10.0)  # 10-second period

    # Sample some points with 1-second gaps
    points = []
    for _ in range(5):
        point = sampler.sample_next_point(time_gap=1.0)
        points.append(point)
"""Constant-velocity Kalman filter with elapsed-time propagation.

State: [cx, cy, w, h, vx, vy, vw, vh]. Measurement: [cx, cy, w, h].

The filter is implemented from the published linear-Gaussian derivation in the
project PRD; no GPL tracker source code is copied. Covariance updates use the
Joseph form and a linear solve instead of an explicit inverse.
"""

from __future__ import annotations

import numpy as np

from shop_core.geometry import box_from_center, validate_box

STATE_DIM = 8
MEASUREMENT_DIM = 4


def transition_matrix(dt_seconds: float) -> np.ndarray:
    if dt_seconds < 0 or not np.isfinite(dt_seconds):
        raise ValueError("elapsed time must be finite and non-negative")
    f = np.eye(STATE_DIM)
    for i in range(4):
        f[i, i + 4] = dt_seconds
    return f


def process_noise(dt_seconds: float, spectral_density: float) -> np.ndarray:
    """White-acceleration model for each position/velocity pair."""
    if dt_seconds < 0 or not np.isfinite(dt_seconds):
        raise ValueError("elapsed time must be finite and non-negative")
    if spectral_density < 0 or not np.isfinite(spectral_density):
        raise ValueError("process noise density must be finite and non-negative")
    q = np.zeros((STATE_DIM, STATE_DIM))
    block = spectral_density * np.array(
        [[dt_seconds**3 / 3.0, dt_seconds**2 / 2.0], [dt_seconds**2 / 2.0, dt_seconds]]
    )
    for i in range(4):
        q[i, i] = block[0, 0]
        q[i, i + 4] = block[0, 1]
        q[i + 4, i] = block[1, 0]
        q[i + 4, i + 4] = block[1, 1]
    return q


class KalmanBoxFilter:
    def __init__(
        self,
        box: tuple[float, float, float, float],
        measurement_variance: float,
        initial_velocity_variance: float = 2500.0,
    ) -> None:
        if measurement_variance <= 0:
            raise ValueError("measurement variance must be positive")
        x1, y1, x2, y2 = validate_box(box)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag(
            [
                measurement_variance,
                measurement_variance,
                measurement_variance,
                measurement_variance,
                initial_velocity_variance,
                initial_velocity_variance,
                initial_velocity_variance,
                initial_velocity_variance,
            ]
        )
        self.R = np.eye(MEASUREMENT_DIM) * measurement_variance
        self.H = np.zeros((MEASUREMENT_DIM, STATE_DIM))
        self.H[:4, :4] = np.eye(MEASUREMENT_DIM)

    @property
    def box(self) -> tuple[float, float, float, float]:
        cx, cy, w, h = self.x[:4]
        return box_from_center(float(cx), float(cy), float(max(w, 1e-3)), float(max(h, 1e-3)))

    def predict(self, dt_seconds: float, spectral_density: float) -> None:
        f = transition_matrix(dt_seconds)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + process_noise(dt_seconds, spectral_density)
        self.x[2] = max(self.x[2], 1e-3)
        self.x[3] = max(self.x[3], 1e-3)
        self.P = symmetrize(self.P)

    def innovation(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.asarray(measurement, dtype=float).reshape(MEASUREMENT_DIM)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        return y, s, z

    def mahalanobis_squared(self, measurement: np.ndarray) -> float:
        y, s, _ = self.innovation(measurement)
        solved = np.linalg.solve(s, y)
        return float(y @ solved)

    def update(self, measurement: np.ndarray) -> None:
        y, s, _ = self.innovation(measurement)
        # K = P H^T S^-1 computed by solving S K^T = (P H^T)^T.
        pht = self.P @ self.H.T
        k = np.linalg.solve(s, pht.T).T
        self.x = self.x + k @ y
        identity = np.eye(STATE_DIM)
        joseph = (identity - k @ self.H) @ self.P @ (identity - k @ self.H).T + k @ self.R @ k.T
        self.P = symmetrize(joseph)
        self.x[2] = max(self.x[2], 1e-3)
        self.x[3] = max(self.x[3], 1e-3)

    def covariance_is_valid(self, tolerance: float = 1e-6) -> bool:
        if not np.allclose(self.P, self.P.T, atol=tolerance):
            return False
        eigenvalues = np.linalg.eigvalsh(self.P)
        return bool(np.all(eigenvalues >= -tolerance))


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0

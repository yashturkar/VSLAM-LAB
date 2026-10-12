"""Pinned BorealHDR Zhang/Kim AE controllers for the offline image emulator.

Based on norlab-ulaval/BorealHDR@042befac24efb256ab2a3c787c758a88a490c290,
scripts/classes/class_auto_exposure_methods.py. The two deliberate adaptations
are documented at their use sites: Zhang weights match the actual input image
size, and Kim's GP has a persistent fixed RNG for reproducible cached runs.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


PINNED_SOURCE = 'norlab-ulaval/BorealHDR@042befac24efb256ab2a3c787c758a88a490c290'
KIM_GP_SEED = 0


class KimController:
    """Entropy-weighted gradient metric and GP/MAXMI exposure search."""

    def __init__(self):
        self.exposures: list[float] = []
        self.metrics: list[float] = []
        self.candidates = np.linspace(0.5, 35, 50)
        self.previous_eta = 1.0
        self.rng = np.random.RandomState(KIM_GP_SEED)
        self.gp = None

    def update(self, bayer: np.ndarray, exposure_ms: float) -> tuple[float, float, str]:
        gray = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2GRAY)
        gray = (gray / 16.0).astype(np.uint8)
        horizontal = cv2.Sobel(gray, ddepth=cv2.CV_16UC1, dx=1, dy=0, ksize=3).ravel()
        vertical = cv2.Sobel(gray, ddepth=cv2.CV_16UC1, dx=0, dy=1, ksize=3).ravel()
        gradient_sq = horizontal ** 2 + vertical ** 2

        histogram, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        probability = histogram / histogram.sum()
        entropy = -probability[gray.ravel()] * np.log2(probability[gray.ravel()])
        sigma = np.std(entropy)
        weights = (1 / sigma) * np.exp((entropy - np.mean(entropy)) ** 2 / (2 * sigma ** 2))
        weights /= np.sum(weights)
        normalized = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
        activation = 2 / (1 + np.exp(-16 * normalized + 4)) - 1
        mask = np.zeros_like(normalized)
        mask[normalized < 0.6] = 1
        n = gradient_sq.shape[0]
        metric = float(np.sum(weights * gradient_sq + activation * mask * weights *
                              (1 / n) * np.sum(gradient_sq)))
        if not np.isfinite(metric):
            raise ValueError('Pinned Kim metric is non-finite on this image')

        self.exposures.insert(0, float(exposure_ms))
        self.metrics.insert(0, metric)
        # Same kernel, observation noise, 10 optimizer restarts, candidate grid,
        # MAXMI acquisition, and five-point sliding window as pinned BorealHDR.
        # A persistent seed is the only GP change: upstream's unseeded restarts
        # would make an identical dataset produce a different cached AE schedule.
        alpha = np.full_like(self.exposures, 0.2)
        self.gp = GaussianProcessRegressor(kernel=1.0 * RBF(), alpha=alpha,
                                           n_restarts_optimizer=10,
                                           random_state=self.rng).fit(
            np.asarray(self.exposures).reshape(-1, 1), np.asarray(self.metrics))
        mean, std = self.gp.predict(self.candidates.reshape(-1, 1), return_std=True)
        variance = std ** 2
        gamma = math.log(2 / 0.25)
        phi = math.sqrt(gamma) * (np.sqrt(variance + self.previous_eta) - math.sqrt(self.previous_eta))
        selected = int(np.argmax(mean + phi))
        self.previous_eta += variance[selected]
        if std[selected] <= 5:
            query = np.linspace(0.5, 35, 1000)
            next_ms = float(query[np.argmax(self.gp.predict(query.reshape(-1, 1)))])
            branch = 'gp_optimal'
        else:
            next_ms = float(self.candidates[selected])
            branch = 'maxmi'
        if len(self.exposures) == 5:
            self.exposures.pop()
            self.metrics.pop()
        return next_ms, metric, branch


class ZhangController:
    """Soft-percentile exposure derivative with classical-50 boundary fallback."""

    def __init__(self, curve: np.ndarray):
        digital = np.linspace(0, 4095, len(curve))
        samples = np.interp(np.linspace(0, 4095, 4097), digital, curve)
        self.icrf_derivative = UnivariateSpline(np.linspace(0, 4095, 4096), np.diff(samples))
        self.weights = None
        self.image_size = None

    @staticmethod
    def _gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kernel_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float32)
        kernel_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)
        x = signal.convolve2d(image, kernel_x, boundary='symm', mode='same')
        y = signal.convolve2d(image, kernel_y, boundary='symm', mode='same')
        return np.float128(x), np.float128(y)

    @staticmethod
    def _weights(size: int) -> np.ndarray:
        # Pinned Zhang hard-codes 1200*1920 here. Boreal's supplied Bayer PNGs
        # are 960*600, so that literal vector cannot multiply their pixels.
        # Keep the same p=0.8, k=5 shape and normalization at actual image size.
        threshold = math.floor(0.8 * size)
        first = np.arange(0, threshold)
        last = np.arange(0, size - threshold)
        weights = np.zeros_like(np.arange(0, size, dtype=np.float32), dtype=np.float32)
        weights[:threshold] = np.sin(np.pi * first / (2 * threshold)) ** 5
        weights[threshold:] = np.sin(np.pi / 2 - np.pi * last / (2 * (size - threshold))) ** 5
        return weights / np.linalg.norm(weights)

    def update(self, bayer: np.ndarray, exposure_ms: float) -> tuple[float, float, str]:
        gray = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2GRAY)
        mean = float(np.mean(gray))
        if mean <= 70 / 256 * 4096 or mean >= 190 / 256 * 4096:
            error = mean - 2048
            next_ms = exposure_ms if abs(error) < 15 else abs(exposure_ms - 0.002 * error)
            branch = 'classical_boundary'
            metric = mean
        else:
            image_x, image_y = self._gradient(gray)
            inverse_derivative = 1.0 / (self.icrf_derivative(gray) * exposure_ms)
            response_x, response_y = self._gradient(inverse_derivative)
            derivative = 2 * (image_x * response_x + image_y * response_y)
            if self.image_size != gray.size:
                self.weights = self._weights(gray.size)
                self.image_size = gray.size
            sorted_derivative = np.sort(derivative.ravel())
            metric = np.sum(self.weights * sorted_derivative)
            next_ms = exposure_ms + 1e-6 * metric
            branch = 'soft_percentile'
        next_ms = float(max(next_ms, 0.02))
        if not np.isfinite(next_ms):
            raise ValueError('Pinned Zhang update is non-finite on this image')
        return next_ms, float(metric), branch

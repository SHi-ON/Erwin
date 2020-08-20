import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr


def base_count(array, width):
    count = np.ceil((array.max() - array.min()) / width)
    count = int(count.item())
    return count


def sqrt_choice(array):
    count = np.ceil(np.sqrt(len(array)))
    return count


def sturge(array):
    count = np.ceil(np.log2(len(array))) + 1
    return count


def rice(array):
    count = np.ceil(2 * np.cbrt(len(array)))
    return count


def doane(array):
    """
    Doane's method

    :param array: data point samples
    :type array: np.ndarray

    :return: bin count
    :rtype: int
    """
    n = len(array)

    m_2 = (1 / n) * np.sum((array - array.mean()) ** 2)
    m_3 = (1 / n) * np.sum((array - array.mean()) ** 3)
    b_1 = m_3 / (m_2 ** (3 / 2))

    sig_b_1 = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))

    count = 1 + np.ceil(np.log2(n)) + np.ceil(np.log2(1 + (np.abs(b_1) / sig_b_1)))
    return count


def scott(array):
    width = 3.49 * array.std() / (len(array) ** (1 / 3))
    return width


def freedman_diaconis(array):
    inter_quartile_range = iqr(array, axis=0)
    width = 2 * inter_quartile_range / (len(array) ** (1 / 3))
    width = width.item()
    return width


def shimazaki_shinomoto(array):
    """
    Shimazaki and Shinomoto's choice. Biased variance is recommended.

    :param array: sample array
    :type array: np.ndarray

    :return: bin-width
    :rtype: float
    """

    def l2_risk_function(points):
        lo = points.min()
        hi = points.max()

        n_min = lo + 1  # min number of splits
        n_max = hi + 1  # max number of splits

        counts = np.arange(n_min, n_max)  # number of bins vector
        widths = (hi - lo) / counts  # width (size) of bins vector

        costs = np.zeros((widths.size, 1))  # cost function values
        for i in range(counts.size):
            edges = np.linspace(lo, hi, counts[i] + 1)
            current_frequency = plt.hist(points, edges)  # number of data points in the current bin
            current_frequency = current_frequency[0]
            costs[i] = (2 * current_frequency.mean() - current_frequency.var()) / (widths[i] ** 2)

        return costs, widths, counts

    cost_function_values, bucket_widths, bucket_counts = l2_risk_function(array)
    optimal_index = np.argmin(cost_function_values)
    optimal_width = bucket_widths[optimal_index]

    return optimal_width

# standard packages
from copy import copy
import math

# 3rd party packages
import numpy as np


class OutlierDetector:
    """
    Wrapper for outlier detection algorithms. Adds generator and outlier algorithm storage for multiple uses. Also adds
    the ability to specify how many sweeps of the data should be performed to detect outliers.
    """
    
    def __init__(self, generator_factory, outlier_algorithm):
        """
        :param generator_factory: (object) produces a generator that partitions the data into subsections
                                  according to the generator's business rules (ex. using a range or number of frames)
        :param outlier_algorithm: (object) contains the method to determine outliers (ex. interquartile range, z-score)
        """
        
        self._generator_factory = generator_factory
        self._outlier_algorithm = outlier_algorithm

    def outliers(self, data, num_of_iterations=None):
        """
        Detects outliers using provided statistical methods.
        
        :param data: (numpy array) time series data organized in columns, where each column is an independent time series and time ordered.
                     Timestamp data should not be included.
        :param num_of_iterations: (int) number of sweeps of the outlier detector over the data. Previously found outliers will be removed so
                                  that new outliers which might have been missed because of other outliers can be detected. If None, the algorithm
                                  will continue to sweep the data until no further outliers are detected.
        
        :return: (boolean array) trues indicate detected outliers.
        """
        if len(data.shape) == 1:
            data = np.reshape(np.array(data), (-1, 1))
        
        outliers = np.zeros(data.shape)
        num_missing_in_frame = np.zeros(data.shape)
        if num_of_iterations is None:
            while True:
                outlier_truth_matrix, num_missing_in_frame = self._outlier_algorithm.outliers(copy(data),
                                                                                              self._generator_factory,
                                                                                              num_missing_in_frame)
                outliers += outlier_truth_matrix
                if np.sum(outlier_truth_matrix) == 0:
                    break
                data[outlier_truth_matrix] = float('nan')
            return outliers.astype(bool)
        else:
            for _ in range(num_of_iterations):
                outlier_truth_matrix, num_missing_in_frame = self._outlier_algorithm.outliers(copy(data),
                                                                                              self._generator_factory,
                                                                                              num_missing_in_frame)
                outliers += outlier_truth_matrix
                if np.sum(outlier_truth_matrix) == 0:
                    break
                data[outlier_truth_matrix] = float('nan')
            return outliers.astype(bool)


class byFrameGeneratorFactory:
    """
    Generator Factory.
    For each datum, generator yields a subset of data that are within 'frame' positions from the datum.
    Since a symmetric frame cannot be built for data at the beginning and end of the dataset, the data
    yielded will be the first/last 2 * frame + 1 measurements.
    """

    def __init__(self, frame=8):
        self._frame = math.ceil(frame)

    def generator(self, data):
        data = copy(data)
        for idx in range(data.shape[0]):
            if self._frame <= idx < (data.shape[0] - self._frame):
                output = data[(idx - self._frame):(idx + self._frame + 1), ]
            elif idx < self._frame:
                output = data[:(2 * self._frame + 1), ]
            else:
                output = data[(-2 * self._frame - 1):, ]
            yield output

    def span(self):
        return self._frame


class InterquartileVarianceAlgorithm:
    """
    Estimated variance based on the interquartile range is used with median of data to determine whether a particular datum is
    an outlier. This particular method is useful for non-stationary time series data.
    """

    def __init__(self, stdev_limit=5, min_num_points=20):
        """
            :param k_factor: (float/int) multiplication factor used to set range for outlier detection, generally agreed to be 1.5
            :param min_num_points: (int) minimum number of data necessary to peform statistics. If this number is larger than the size
                                   of the frame, the frame size will be used.
        """
        
        self._stdev_limit = stdev_limit
        self._min_num_points = min_num_points  # minimum number of data necessary for statistics

    def outliers(self, data, generator_factory, num_missing_data_in_frame):
        """
        Creates a boolean matrix of detected outliers.
        
        :param data: (array like) Time-series data which has been sorted by time. Non-data columns (ex. time) should not be included.
                     Each column of data is treated as independent. 
        :param generator_factory: (class object) a factory which creates a data generator used to break up the data for statistical analysis.
                                  Should have .outlier(data) method.
        :param num_missing_data_in_frame: (integer array) memoization array tracking the number of nan values within a given generator frame.
                                          If the number of nan values has not changed from a previous iteration, the statistics will be skipped
                                          (as nothing has changed). Should be all zeros when beginning method.
        
        :return: (boolean array, integer array) trues indicate detected outliers in boolean array. Integer array is num_missing_data_in_frame.
        """
        
        span = generator_factory.span()  # position of the center point yielded by the generator
        if len(data.shape) == 1:
            data = np.reshape(np.array(data), (-1, 1))
        outliers = np.zeros(data.shape)
        generator = generator_factory.generator(data)

        min_num_points = min(2 * span + 1, self._min_num_points)  # minimum number of non-NaN data to do statistics
        for count, frame in enumerate(generator):
            new_num_missing_data = [sum(~np.isnan(frame[:, i])) for i in range(frame.shape[1])]
            need_recalc = [True if new_num_missing_data[i] != num_missing_data_in_frame[count, i] else False
                           for i in range(len(new_num_missing_data))]
            row_outliers = [is_outlier(frame[:, i], data[count, i], self._stdev_limit, min_num_points)
                            if need_recalc[i] else False
                            for i in range(frame.shape[1])]
            if not np.array_equal(num_missing_data_in_frame[count], new_num_missing_data):
                num_missing_data_in_frame[count] = new_num_missing_data
            outliers[count] = row_outliers
        return outliers.astype(bool), num_missing_data_in_frame


def is_outlier(col, data_point, std_dev_limit, min_num_points):
    """
    Estimates the variance of the col data using the interquartile range. Outputs whether data_point is outside
    median +/- std_dev_limit * estimated variance. min_num_points is used to make sure there is enough data to 
    result in good statistics.
    
    :param col: (array-like) data in question.
    :param data_point: (float) datum to evaluate as outlier.
    :param std_dev_limit: (int) the number of standard deviations away from the median a datum must be to be considered
                          an outlier.
    :param min_num_points: (int) the minimum number of non-nan data in col required to perform interquartile statistics.
                           Not enough data results in a False output.
    
    :return: (bool) whether the data_point is an outlier (if there is enough data in col).
    """
    
    q25 = np.nanpercentile(col.astype(float), 25) if sum(~np.isnan(col.astype(float))) >= min_num_points else float('nan')
    q75 = np.nanpercentile(col.astype(float), 75) if sum(~np.isnan(col.astype(float))) >= min_num_points else float('nan')
    iqr_stdev = (abs(q75 - q25) / 1.35) if int(sum(np.isnan([q75, q25]))) == 0 else float('nan')
    cut_off = iqr_stdev * std_dev_limit
    return (False if math.isnan(cut_off) | math.isnan(data_point) |
                        ((np.nanmedian(col) - cut_off) < data_point < (np.nanmedian(col) + cut_off)) else True)

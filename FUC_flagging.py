import numpy as np


def FUC_curve_generator(FUC_data, min_time_breakpoint, ignoreFirst = False):
    """
    Isolates and returns FUC curve indicies from a time series of FUC curves.
    
    :param FUC_data: chronologically ordered numpy array with [datetime, FUC xCO2 data] columns. Datetime column
                     should be populated with datetime objects
    :param min_time_breakpoint: numpy timedelta64 object determinining the minimum time between two measurements
                                which constitutes a new FUC curve.
    :param ignoreFirst: Do not return the first FUC curve. This is for when it is likely the data starts with a
                        partial FUC curve (preferably one that has already been flagged), so flagging should not
                        be performed.
    
    :return: yields indicies of isolated FUC curve
    """
    
    i = 0  # lagging frame
    j = 1  # leading frame
    while j < FUC_data.shape[0]:
        if abs(FUC_data[j] - FUC_data[j - 1]) > min_time_breakpoint:
            if not ignoreFirst:
                yield i, j - 1
                i = j
                j += 1
            else:
                ignoreFirst = False
                i = j
                j += 1
        else:
            j += 1
    yield i, j - 1


def FUC_tests(FUC_curve, span_co2: float, min_num_points: int, max_diff_from_span: float, max_st_dev: float) -> bool:
    """
    Return flag and (flag note) for an FUC curve. Flags are determined by 3 tests, any positive will result in a flag
    of True (bad). The last three measurements of the FUC curve are post calibrated, so they are ignored and the last
    5 points of the FUC curve ([-8:-3]) before post calibration are used for tests 2 and 3.
		Test 1: whether the FUC curve has greater than min_num_points
		Test 2: whether the difference between the average of the FUC curve at positions [-8:-3] and the span_co2 is
		        less than max_diff_from_span
		Test 3: whether the difference between the standard deviation of the FUC curve at positions [-8:-3] is less
		        than max_st_dev
	
	:param FUC_curve: (numpy array) An isolated FUC curve with column 0 being datetime objects and column 1 being
	                  xCO2 dry measurements.
	:param span_co2: (float) The span gas concentration, in ppm, which the FUC curve should approach then be calibrated to.
	:param min_num_points: (int) the minimum number of points necessary for a FUC curve to be considered viable
	:param max_diff_from_span: (float) The maximum allowable difference between the average of the last 5 points of the
	                           FUC curve (before post cal data) and the span gas concentration.
	:param max_h2o_press: (float) The maximum allowable H2O partial pressure in mbar.
    :param max_st_dev: (float) The maximum allowable standard deviation of the last 5 points of the FUC curve
	                   (before post cal data)
	
	:return: (bool) flag of FUC curve. False is good data, True is bad data.
    """

    if FUC_curve.shape[0] < min_num_points:
        return True
    last_FUC_measurements = FUC_curve[-8:-3]
    if abs(span_co2 - np.mean(last_FUC_measurements)) > max_diff_from_span:
        return True
    if np.std(last_FUC_measurements) > max_st_dev:
        return True
    return False


def qc_FUC_timeseries(FUC_time_series, min_time_breakpoint, span_co2, min_num_points=10, max_diff_from_span=5, max_st_dev=2.25, ignoreFirst=False):
    """
    Isolates each FUC curve of FUC_time_series and applies a flag determined by the FUC_tests function. Outputs an array
    of [timestamp, flag] for each identified FUC curve. If ignoreFirst is true, then skips the first FUC curve identified
    (in case that curve is incomplete).
	
    :param FUC_time_series: (numpy array) chronologically ordered numpy array with [datetime, FUC xCO2 data] columns
	                        (multiple FUC curves expected). Datetime column should be populated with datetime objects.
    :param min_time_breakpoint: numpy timedelta64 object determining the minimum time between two measurements
                                which constitutes a new FUC curve.
	:param min_num_points: (int) the minimum number of points necessary for a FUC curve to be considered viable
	:param max_diff_from_span: (float) The maximum allowable difference between the average of the last 5 points of
	                            the FUC curve (before post cal data) and the span gas concentration.
	:param max_st_dev: (float) The maximum allowable standard deviation of the last 5 points of the FUC curve
	                   (before post cal data)
    :param max_h2o_press: (float) The maximum allowable H2O partial pressure in mbar.
    :param ignoreFirst: Do not return the first FUC curve. This is for when it is likely the data starts with a
                        partial FUC curve (preferably one that has already been flagged), so flagging should not
                        be performed.

	:return: Numpy array of [timestamp, FUC flag] for each unique FUC curve identified (not per datetime in FUC_time_series).
	         Timestamp is first datetime of FUC curve.
    """
    
    generator = FUC_curve_generator(FUC_time_series[:, 0], min_time_breakpoint, ignoreFirst)
    output = []
    for curve_start, curve_end in generator:
        if curve_end - curve_start == 0:
            # failsafe if curve is ever only one point, since a slice must have its first and second indicies be different
            curve_end += 1
        timestamp = FUC_time_series[curve_start, 0]
        flag = FUC_tests(FUC_time_series[curve_start:curve_end, 1], span_co2, min_num_points, max_diff_from_span, max_st_dev)
        output.append([timestamp, flag])
    return np.array(output)


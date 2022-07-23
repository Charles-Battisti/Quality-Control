# Quality-Control
Different methods to quality control data.

FUC_flagging isolates and quality controls calibration data from an atmospheric CO2 data stream.

memoized_outlier_detector is a generalized outlier detector for time series data. An outlier is identified using the median and standard deviation of the surrounding data. The method computes standard deviation using the interquartile range rather than the mathematical standard deviation, which is more stable in the presence of potential extreme outliers. Unless otherwise specified, the detector will repeatedly scan the data until no more outliers are detected. To decrease running time, the detector is memoized.

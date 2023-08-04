# Visibility Graph Metrics for Estimating HRV Features
This repository holds my workspace for exploring the ability of several visibility graph metrics to estimate heart-rate-variability features and measures.
***This work is done as part of a student research assistant position at the Robust Data Science Group at the Technical University of Darmstadt.***

## Contents
* vg_metric.py: contains a class for computing visibility graph metrics segment-wise for a given time series (RR-series)
* compute_correlation.py: a script to compute the acceleration/deceleration capacity for ECG records and correlate those with visibility graph metrics
* read_data.py: helper class to load long-term ECG recordings
* compute_r_peaks.py: helper script to compute R-peaks of ECG recordings using visibility graph based R-peak detectors



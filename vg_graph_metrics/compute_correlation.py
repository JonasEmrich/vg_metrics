import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy import signal
import pandas as pd
import multiprocessing as mp
import os

from vg_metric import VisGraphMetric
from tqdm import tqdm
from read_data import ReadMainzData 

path_output = "/home/jemrich/vg_graph_metrics/capacity_values_agg.csv"
path_rpeaks = "/home/jemrich/mdata_rpeaks/"
path_data = "/home/jemrich/mdata/"

# %% functions to compute metrics
def segmentwise_ACDC(rr, fs, skip_outlier=True):
   """computes the 4 sample long segments used for calculating the acceleration and deceleration capacity."""
   rr = rr/fs*1000
   N = len(rr)

   values = []
   labels = []
   indices = []

   for i in range(2,N-1):
      #compute AC/DC formula if not outlier, i.e., in 5% bounds
      if skip_outlier and not 1.05 * rr[i-1] > rr[i] > 0.95 * rr[i-1]:
        continue

      values.append((rr[i]+rr[i+1]-rr[i-1]-rr[i-2])/4)
      indices.append(i)

      # label as AC or DC
      if rr[i] <= rr[i-1]:
         labels.append("AC")
      elif rr[i] > rr[i-1]:
         labels.append("DC")

   return np.array(values), np.array(labels), np.array(indices)

def calc_correlations_segmentwise(values, indices, labels, rr, record, VGM, metrics):
      data = []

      m = VGM.calc_metric(rr, metrics=metrics, quiet=True)
      for metric_name, (m, ix_m) in m.items():
         _values = values.copy()
         _indices = indices.copy()
         _labels = labels.copy() 

         ix_m += VGM.beats_per_window//2 # shift to middle of window
         # get common indices
         idx = np.intersect1d(_indices, ix_m)

         _values = _values[np.in1d(_indices, idx)]
         m = m[np.in1d(ix_m, idx)]
         _labels = _labels[np.in1d(_indices, idx)]
            
         AC_true = _values[_labels == "AC"]
         AC_pred = m[_labels == "AC"]
         DC_true = _values[_labels == "DC"]
         DC_pred = m[_labels == "DC"]

         AC = np.corrcoef(AC_true, AC_pred, rowvar=False)[0,1]
         DC = np.corrcoef(DC_true, DC_pred, rowvar=False)[0,1]
         ALL = np.corrcoef(_values, m, rowvar=False)[0,1]

         #if np.any(np.array([ALL, AC, DC]) > 0.7):
         #    print(f"{record}: {metric_name}_{str(VGM.directed)}_{str(VGM.edge_weight)}_{VGM.beats_per_window} \t AC: {AC} \t DC: {DC} \t ALL: {ALL}")

         data.append([record, metric_name, ALL, AC, DC, VGM.beats_per_window, str(VGM.edge_weight), str(VGM.directed), freq_domain])  

      df = pd.DataFrame(data, columns=['record', 'name', 'ALL', 'AC', 'DC', 'beats_per_window', 'edge_weight', 'direction', 'freq_domain'])
      df.to_csv(path_output, index=False, mode='a', header=False)   


def calc_correlations_aggregate(values, indices, labels, rr, record, VGM, metrics):
      data = []

      # # calculate standard acceleration/deceleration capacity
      # AC = np.mean(values[labels == "AC"])   
      # DC = np.mean(values[labels == "DC"])
      # data.append([record, "standard", AC, DC, None, None, None, None]) 

      m = VGM.calc_metric(rr, metrics=metrics, quiet=True, skip_outlier=True)
      for metric_name, (m, ix_m) in m.items():
         non_outlier=(m != 0) & (ix_m != 0)
         m = m[non_outlier]
         ix_m = ix_m[non_outlier]

         _indices = indices.copy()
         _labels = labels.copy() 

         ix_m += VGM.beats_per_window//2 # shift to middle of window
         # get common indices
         idx = np.intersect1d(_indices, ix_m)

         m = m[np.in1d(ix_m, idx)]
         _labels = _labels[np.in1d(_indices, idx)]
            
         AC_pred = m[_labels == "AC"]
         DC_pred = m[_labels == "DC"]

         for _name, _agg in dict(skip_outlier_mean=np.mean, skip_outlier_median=np.median).items():
            data.append([record, metric_name+"_"+_name, _agg(AC_pred), _agg(DC_pred), VGM.beats_per_window, str(VGM.edge_weight), str(VGM.directed), freq_domain]) 

      df = pd.DataFrame(data, columns=['record', 'name', 'AC', 'DC', 'beats_per_window', 'edge_weight', 'direction', 'freq_domain'])
      df.to_csv(path_output, index=False, mode='a', header=False)   

# %% functions to execute
def correlations_segmentwise_per_record(record):
   r_peaks = np.load(path_rpeaks+record+".npy")
   rr = np.diff(r_peaks)
   fs = 256
   # calculate AC/DC
   values, labels, indices = segmentwise_ACDC(rr, fs)
   # calculate graph metrics
   for bpw in beats_per_window:
      for edge_weight in edge_weights:
         for direction in directions:
            VGM = VisGraphMetric(edge_weight=edge_weight, direction=direction, freq_domain=freq_domain, beats_per_window=bpw, beats_per_step=1)
            calc_correlations_segmentwise(values, indices, labels, rr, record, VGM, metrics)

def correlations_aggregate_per_record(record):
   r_peaks = np.load(path_rpeaks+record+".npy")
   rr = np.diff(r_peaks)
   fs = 256
   # calculate AC/DC
   values, labels, indices = segmentwise_ACDC(rr, fs)
   # calculate graph metrics
   for bpw in beats_per_window:
      for edge_weight in edge_weights:
         for direction in directions:
            VGM = VisGraphMetric(edge_weight=edge_weight, direction=direction, freq_domain=freq_domain, beats_per_window=bpw, beats_per_step=1)
            calc_correlations_aggregate(values, indices, labels, rr, record, VGM, metrics)

# %% RUN 
beats_per_window = [4]#,8,16]
freq_domain = False
directions = ['left_to_right']#None, 
edge_weights = ['v_distance']#'slope', 'angle', ]#None,'abs_slope', 'abs_angle', 'distance', 'sq_distance', 'abs_v_distance', 'h_distance', 'abs_h_distance']
metrics=['shortest_path_length_left_to_right']#'minimum_cut_value_left_to_right', 'maximum_flow_value_left_to_right', 'shortest_path_length_left_to_right', 'dag_longest_path_length']
# VGM = VisGraphMetric(edge_weight="slope", direction=None, freq_domain=freq_domain, beats_per_window=beats_per_window, beats_per_step=1)

print("Number of processors: ", mp.cpu_count())

files = [os.path.splitext(filename)[0] for filename in os.listdir(path_rpeaks)]
pool = mp.Pool(processes=mp.cpu_count())
for result in tqdm(pool.imap(correlations_aggregate_per_record, files), total=len(files), desc="records"):
    pass




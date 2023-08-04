import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy import signal
import pandas as pd
import multiprocessing as mp

from vg_metric import VisGraphMetric
from tqdm import tqdm
from read_data import ReadMainzData 

path_output = "/home/jemrich/vg_graph_metrics/correlation.csv"
path_rpeaks = "/home/jemrich/mdata_rpeaks/"
path_data = "/home/jemrich/mdata/"

# %% functions
def segmentwise_ACDC(rr, fs):
   rr = rr/fs*1000
   N = len(rr)

   values = []
   labels = []
   indices = []

   for i in range(2,N-1):
      # compute AC/DC formula if not outlier, i.e., in 5% bounds
      if not 1.05 * rr[i-1] > rr[i] > 0.95 * rr[i-1]:
         values.append(np.nan)
         labels.append("outlier")
         indices.append(i)
         continue

      values.append((rr[i]+rr[i+1]-rr[i-1]-rr[i-2])/4)
      indices.append(i)

      # label as AC or DC
      if rr[i] <= rr[i-1]:
         labels.append("AC")
      elif rr[i] > rr[i-1]:
         labels.append("DC")

   return np.array(values), np.array(labels), np.array(indices)

def calc_correlations(values, indices, labels, rr, record, VGM, metrics):
        data = []

        m = VGM.calc_metric(rr, metrics=metrics, quiet=True)
        for metric_name, (m, ix_m) in m.items():
            ix_m += beats_per_window//2 # shift to middle of window
            # get common indices
            idx = np.intersect1d(indices, ix_m)

            values = values[np.in1d(indices, idx)]
            m = m[np.in1d(ix_m, idx)]
            labels = labels[np.in1d(indices, idx)]
            
            AC_true = values[labels == "AC"]
            AC_pred = m[labels == "AC"]
            DC_true = values[labels == "DC"]
            DC_pred = m[labels == "DC"]

            AC = np.corrcoef(AC_true, AC_pred, rowvar=False)[0,1]
            DC = np.corrcoef(DC_true, DC_pred, rowvar=False)[0,1]
            ALL = np.corrcoef(values, m, rowvar=False)[0,1]

            #if np.any(np.array([ALL, AC, DC]) > 0.7):
            #    print(f"{record}: {metric_name}_{str(VGM.directed)}_{str(VGM.edge_weight)}_{VGM.beats_per_window} \t AC: {AC} \t DC: {DC} \t ALL: {ALL}")

            data.append([record, metric_name, ALL, AC, DC, VGM.beats_per_window, str(VGM.edge_weight), str(VGM.directed), freq_domain])  

        df = pd.DataFrame(data, columns=['record', 'name', 'ALL', 'AC', 'DC', 'beats_per_window', 'edge_weight', 'direction', 'freq_domain'])
        df.to_csv(path_output, index=False, mode='a', header=False)   

# %%
beats_per_window = 4
freq_domain = False
directions = [None, 'left_to_right']
edge_weights = [None, 'slope', 'angle', 'v_distance']#'abs_slope', 'abs_angle', 'distance', 'sq_distance', 'abs_v_distance', 'h_distance', 'abs_h_distance' 
metrics=['minimum_cut_value_left_to_right', 'maximum_flow_value_left_to_right', 'shortest_path_length_left_to_right', 'dag_longest_path_length']
# VGM = VisGraphMetric(edge_weight="slope", direction=None, freq_domain=freq_domain, beats_per_window=beats_per_window, beats_per_step=1)

def correlation_per_record(record):
    r_peaks = np.load(path_rpeaks+record+".npy")
    rr = np.diff(r_peaks)
    fs = wfdb.rdheader(path_data+record).fs

    # calculate AC/DC
    values, labels, indices = segmentwise_ACDC(rr, fs)
    values = values[labels != "outlier"]
    indices = indices[labels != "outlier"]
    labels = labels[labels != "outlier"]

    # calculate graph metrics
    for edge_weight in edge_weights:
        for direction in directions:
            VGM = VisGraphMetric(edge_weight=edge_weight, direction=direction, freq_domain=freq_domain, beats_per_window=beats_per_window, beats_per_step=1)
            #metrics = VGM.get_available_metrics()
            calc_correlations(values, indices, labels, rr, record, VGM, metrics)

RMD = ReadMainzData()
# for record in tqdm(RMD.get_record_list(num=40), desc="records"):
#     correlation_per_record(record, metrics, edge_weights, directions, freq_domain, beats_per_window)
#     break


print("Number of processors: ", mp.cpu_count())
pool = mp.Pool(processes=mp.cpu_count())
for result in tqdm(pool.imap(correlation_per_record, RMD.get_record_list(start=0, num=140)), total=140, desc="records"):
    pass



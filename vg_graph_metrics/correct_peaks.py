import numpy as np
from tqdm import tqdm
from read_data import ReadMainzData 
from vg_beat_detectors import FastNVG
import multiprocessing as mp


import pandas as pd
import robustsp as rsp
import numpy as np
import math
import statistics
from os import listdir

path_rpeaks = "/home/jemrich/mdata_rpeaks/"
path_output = "/home/jemrich/mdata_rr_cleaned/"


def cleaninsegments(record):
    size = 1000
    RRseries = np.diff(np.load(path_rpeaks+record))


    spt = np.array_split(RRseries,math.ceil(len(RRseries)/size ))
    RRcleaned = []

    for segment in spt:
        med = statistics.median(segment)
        segment = np.array(segment)
        try:
            cleaned = rsp.arma_est_bip_mm(segment[::-1] - med, p=2, q=0)
            cleaned = cleaned['cleaned_signal'][::-1] + med
            cleaned.tolist()
        except:
            cleaned = segment
            cleaned.tolist()

        RRcleaned = [*RRcleaned, *cleaned]

    np.save(path_output+record, RRcleaned)




pool = mp.Pool(processes=mp.cpu_count())
files = listdir(path_rpeaks)
for result in tqdm(pool.imap(cleaninsegments, files), total=len(files), desc="records"):
    break
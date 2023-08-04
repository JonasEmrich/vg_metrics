import numpy as np
from tqdm import tqdm
from read_data import ReadMainzData 
from vg_beat_detectors import FastNVG
import multiprocessing as mp

RMD = ReadMainzData()

def calculate_rpeaks(record):
    t, x, fs = RMD.load_record(record)
    detector = FastNVG(sampling_frequency=fs)
    rpeaks = detector.find_peaks(x)

    np.save("/home/jemrich/mdata_rpeaks/"+record, rpeaks)

pool = mp.Pool(processes=mp.cpu_count())
for result in tqdm(pool.imap(calculate_rpeaks, RMD.get_record_list(start=40, num=100)), total=100, desc="records"):
    pass
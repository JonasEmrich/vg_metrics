import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import wfdb


class ReadMainzData():
    """
    This class simplifies the usage with the long term ECG data from University of Mainz provided in the .hea format.
    Author: Jonas Emrich
    """
    def __init__(self, priority_sorted=False, only_priority=None, path_to_dataset="/home/jemrich/mdata/",
                 path_to_priority_xlsx="/home/jemrich/vg_graph_metrics/Priorisierung_LZEKG_Export_TU_Darmstadt.xlsx"):
        
        self.path_to_dataset = path_to_dataset
        self.path_to_priority_xlsx = path_to_priority_xlsx
        df = pd.read_excel(path_to_priority_xlsx)

        # sorts the files after priority
        if priority_sorted:
            df = df.sort_values("priority")

        # Create new empty list
        self.records = []

        # Fill the list with all file names without hea
        for record in df['PseudoHEAFilename']:
            self.records.append(record.split('.HEA')[0])
        # Add list (of filenames without hea) to the dataframe
        df["FilenamesWithoutHea"] = self.records

        # If you want all files with a certain priority
        if only_priority is not None:
            self.records = df[df["priority"] == only_priority]["FilenamesWithoutHea"]


    def get_record_list(self, num=None):
        """ Returns a list of all records, or if 'num' is specified, only the prvided number of records. """
        return self.records[:num]
    
    def load_record(self, name, sampfrom=0, sampto=None):
        """ Returns time, signal, and sampling frequency for a given record 'name'. If 'sampfrom' or 'sampto' are given the signal is truncated accordingly."""
        if name not in self.records:
            raise ValueError(f"Invalid record title: {name}.")

        # Create the path to the individual file
        filename = self.path_to_dataset + name
        signal, fields = wfdb.rdsamp(filename, channels=[0], sampfrom=sampfrom, sampto=sampto)

        fs = fields['fs']
        time = np.arange(sampfrom/fs, (sampfrom+len(signal))/ fs, 1 / fs)
        signal = signal.reshape((len(signal)))

        return time, signal, fs








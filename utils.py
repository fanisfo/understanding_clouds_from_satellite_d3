import numpy as np

def pixel_encoder(fl_output):
    fl_array = fl_output.numpy()
    fl_array_diff = np.diff(fl_array)
    start_indices = list(np.where(fl_array_diff == 1) +1)
    end_indices = list(np.where(fl_array_diff == -1) +1)
    if min(end_indices) < min(start_indices):
        start_indices.insert(0,0)
    if max(end_indices) < max(start_indices):
        end_indices.append(len(fl_array_diff))
    result = []
    for start, end in zip(start_indices, end_indices):
        result.extend([start, end-start+1])
    return result
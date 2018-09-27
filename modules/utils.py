#!/usr/bin/env python
import numpy as np
import pandas as pd

def binary_target(target):
  if target <> 0:
    return 1
  else:
    return 0

def replaceNanWithAverage(data):
  col_mean = np.nanmean(data, axis=0)
  idxs = np.where(pd.isnull(data))
  data[idxs] = np.take(col_mean, idxs[1])
  return data

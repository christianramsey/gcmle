# bucketise_column
import numpy as np; import tensorflow as tf;
import tensorflow.contrib.layers as tflayers


x = tflayers.real_valued_column(np.linspace(0, 100, 101))
buckets = np.linspace(0, 100, 5).tolist()

x2 = tflayers.bucketized_column(x, buckets)


import pandas as pd

print(x2)

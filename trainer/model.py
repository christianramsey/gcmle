import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
import tensorflow.contrib.metrics as tfmetrics



# our data doesn't have column names or an explicit y,
# we will define the header here to get ready to ingest the data
CSV_COLUMNS  = \
('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay,carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'


#  tf csv reader wants us to give our default values for the data in case they are empty and to get the column data type
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0], ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]


# a function to read the training and eval dataset, which will give us batch_size examples each time,
# which goes through the dataset num_training_epochs times. the modekeys allows us to make conditional since
# we only w`ant to run through the code once in evaluation
def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.EVAL, batch_size=512, num_training_epochs=10):
    # the actual input function passed to TensorFlow
    def _input_fn():
        num_epochs = num_training_epochs if mode == tf.contrib.learn.ModeKeys.TRAIN else 1

        # could be a path to one file or a file pattern. (shuffling data means we lower our chance of leaving a batch for a slow worker
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(
            input_file_names, num_epochs=num_epochs, shuffle=True)

        # read CSV
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(filename_queue, num_records=batch_size)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL_COLUMN)
        print(features, label)
        return features, label

    return _input_fn


def get_features():
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay,dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    sparse = {
      'carrier': tflayers.sparse_column_with_keys('carrier',
                 keys='AS,VX,F9,UA,US,WN,HA,EV,MQ,DL,OO,B6,NK,AA'.split(',')),

      'origin' : tflayers.sparse_column_with_hash_bucket('origin',
                 hash_bucket_size=1000), # FIXME

      'dest'   : tflayers.sparse_column_with_hash_bucket('dest',
                 hash_bucket_size=1000)  # FIXME
    }

    latbuckets = np.linspace(20.0, 50.0, nbuckets).tolist()  # USA
    lonbuckets = np.linspace(-120.0, -70.0, nbuckets).tolist()  # USA
    disc = {}
    disc.update({
        'd_{}'.format(key): tflayers.bucketized_column(real[key], latbuckets) \
        for key in ['dep_lat', 'arr_lat']
    })
    disc.update({
        'd_{}'.format(key): tflayers.bucketized_column(real[key], lonbuckets) \
        for key in ['dep_lon', 'arr_lon']
    })
    return real, sparse


def create_embed(sparse_col):
    dim = 5 # default
    if hasattr(sparse_col, 'bucket_size'):
       nbins = sparse_col.bucket_size
       if nbins is not None:
        print(nbins)
        dim = 1 + int(round(np.log2(nbins) ) )
    return tflayers.embedding_column(sparse_col, dimension=dim)

def linear_model(output_dir):
    real, sparse = get_features()
    all = {}
    all.update(real)
    all.update(sparse)
    estimator = tflearn.LinearClassifier(model_dir=output_dir, feature_columns=all.values())
    estimator.params["head"]._thresholds = [0.7]
    return estimator

def dnn_model(output_dir):
    real, sparse = get_features()
    all = {}
    all.update(real)
    embed = {
       colname : create_embed(col) \
          for colname, col in sparse.items()
    }
    all.update(embed)

    estimator = tflearn.DNNClassifier(model_dir=output_dir,
                                      feature_columns=all.values(),
                                      hidden_units=[64, 16, 4])
    estimator.params["head"]._thresholds = [0.7]
    return estimator



def serving_input_fn():
    real, sparse = get_features()

    feature_placeholders = {
        key: tf.placeholder(tf.float32, [None]) \
        for key in real.keys()
    }
    feature_placeholders.update({
        key: tf.placeholder(tf.string, [None]) \
        for key in sparse.keys()
    })

    features = {
        # tf.expand_dims will insert a dimension 1 into tensor shape
        # This will make the input tensor a batch of 1
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tflearn.utils.input_fn_utils.InputFnOps(
        features,
        None,
        feature_placeholders)


def serving_input_fn():
    real, sparse = get_features()

    feature_placeholders = {
        key: tf.placeholder(tf.float32, [None]) \
        for key in real.keys()
    }
    feature_placeholders.update({
        key: tf.placeholder(tf.string, [None]) \
        for key in sparse.keys()
    })

    features = {
        # tf.expand_dims will insert a dimension 1 into tensor shape
        # This will make the input tensor a batch of 1
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    # only supported for serving input fn
    return tflearn.utils.input_fn_utils.InputFnOps(
        features,
        None,
        feature_placeholders)

def my_rmse(predictions, labels, **args):
  prob_ontime = predictions[:,1]
  print('--------------------------prob_ontime: ',prob_ontime)
  return tfmetrics.streaming_root_mean_squared_error(prob_ontime,
                       labels, **args)


def make_experiment_fn(traindata, evaldata, **args):

  def _experiment_fn(output_dir):

    return tflearn.Experiment(
        dnn_model(output_dir),
        train_input_fn=read_dataset(traindata,
        mode=tf.contrib.learn.ModeKeys.TRAIN),
        eval_input_fn=read_dataset(evaldata),
        eval_metrics={
            'rmse': tflearn.MetricSpec(metric_fn=my_rmse,
                                       prediction_key='probabilities')
        },
        export_strategies=[saved_model_export_utils.make_export_strategy(
            serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )],
        **args
    )
  return _experiment_fn
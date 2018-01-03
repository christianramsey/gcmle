import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.learn as tflearn

# our data doesn't have column names or an explicit y,
# we will define the header here to get ready to ingest the data
CSV_COLUMNS  = \
('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + \
 'carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'


#  tf csv reader wants us to give our default values for the data in case they are empty and to get the column data type
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]


# a function to read the training and eval dataset, which will give us batch_size examples each time,
# which goes through the dataset num_training_epochs times. the modekeys allows us to make conditional since
# we only want to run through the code once in evaluation
def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.EVAL,
                 batch_size=512, num_training_epochs=10):
    num_epochs = num_training_epochs if mode == tf.contrib.learn.ModeKeys.TRAIN else 1
    # could be a path to one file or a file pattern. (shuffling data means we lower our chance of leaving a batch for a slow worker
    input_file_names = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=num_epochs, shuffle=True)

    # read CSV from tf
    reader = tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records=batch_size)
    value_column = tf.expand_dims(value, -1)
    columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    label = features.pop(LABEL_COLUMN)

    return features, label


def get_features():
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' +
             ',dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    sparse = {
      'carrier': tflayers.sparse_column_with_keys('carrier',
                 keys='AS,VX,F9,UA,US,WN,HA,EV,MQ,DL,OO,B6,NK,AA'.split(',')),

      'origin' : tflayers.sparse_column_with_hash_bucket('origin',
                 hash_bucket_size=1000), # FIXME

      'dest'   : tflayers.sparse_column_with_hash_bucket('dest',
                 hash_bucket_size=1000)  # FIXME
    }
    return real, sparse


def linear_model(output_dir):
    real, sparse = get_features()
    all = {}
    all.update(real)
    all.update(sparse)
    return tflearn.LinearClassifier(model_dir=output_dir,
                                    feature_columns=all.values())
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

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
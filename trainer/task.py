
import tensorflow as tf
import model
import argparse

from tensorflow.contrib.learn.python.learn import learn_runner





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--traindata',
      help='Training data file(s)',
      required=True
  )

  # parse args
  args = parser.parse_args()
  arguments = args.__dict__
  traindata = arguments.pop('traindata')

  feats, label = model.read_dataset(traindata)
  avg = tf.reduce_mean(label)
  print avg


# run
tf.logging.set_verbosity(tf.logging.INFO)
learn_runner.run(model.make_experiment_fn(**arguments), output_dir)
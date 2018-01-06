
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
    parser.add_argument(
      '--output_dir',
      help='Output directory',
      required=True
    )
    parser.add_argument(
      '--evaldata',
      help='Eval data file(s)',
      required=True
    )
    parser.add_argument(
      '--job-dir',
      help='specify job dir for ml engine job',
      default='./junk'
    )
    # parse args
    args = parser.parse_args()
    arguments  = args.__dict__

    arguments.pop('job-dir', None)
    arguments.pop('job_dir', None)
    output_dir = arguments.pop('output_dir')

    # job_dir = arguments['job-dir']

    # run
    tf.logging.set_verbosity(tf.logging.INFO)
    learn_runner.run(model.make_experiment_fn(**arguments), output_dir)
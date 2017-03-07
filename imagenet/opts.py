
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## dataset parameter
tf.app.flags.DEFINE_integer( 'num_class',             1000,    "Number of classes in dataset")
tf.app.flags.DEFINE_integer( 'num_example_train',  1281167,    "Number of training data (images per epoch)")
tf.app.flags.DEFINE_integer( 'num_example_test',   10000,    "Number of test data")
tf.app.flags.DEFINE_integer( 'num_example_eval',   50000,    "Number of test data")
tf.app.flags.DEFINE_integer( 'image_height',          32,    "height of original image") 
tf.app.flags.DEFINE_integer( 'image_width',           32,    "width of original image") 
tf.app.flags.DEFINE_integer( 'image_channel',          3,    "channel of original image") 

## train parameter
tf.app.flags.DEFINE_string(   'test_dir',            './test_tfrecord',   "Directory for test data")
tf.app.flags.DEFINE_string(   'eval_dir',   './anno_full/unsupervised',   "Directory for test data")
tf.app.flags.DEFINE_string(   'eval_save',          './anno_full/eval',   "Directory for test data")
#tf.app.flags.DEFINE_string(   'train_dir',    '/home/user/dataset/ILSVRC2015/Data/CLS-LOC/train',   "Directory of training data")
tf.app.flags.DEFINE_string(   'train_dir',    './text',   "Directory of training data")
tf.app.flags.DEFINE_integer(  'crop_h',                            224,   "Crop size (height)") 
tf.app.flags.DEFINE_integer(  'crop_w',                            224,   "Crop size (width)") 
tf.app.flags.DEFINE_integer(  'crop_ch',                             3,   "Crop size (channel)") 
tf.app.flags.DEFINE_boolean(  'shuffle',                          True,   "Whether to shuffle the batch on training")
tf.app.flags.DEFINE_integer(  'pad_h',                               4,   "Pad each side of image (height)") 
tf.app.flags.DEFINE_integer(  'pad_w',                               4,   "Pad each side of image (width)") 
tf.app.flags.DEFINE_boolean(  'nesterov',                         True,   "Whether to use nesterov on training")

## hyper-parameter
tf.app.flags.DEFINE_integer( 'batch_size',        64,    "Number of mini-batch")
tf.app.flags.DEFINE_float(   'momentum',         0.9,    "Momentum for SGD")
tf.app.flags.DEFINE_float(   'weight_decay',  0.0002,    "Weight decay")
tf.app.flags.DEFINE_float(   'LR',               0.1,    "Base learning rate")
tf.app.flags.DEFINE_float(   'LR_decay',         0.1,    "Learning rate decaying factor")
tf.app.flags.DEFINE_integer( 'LR_step',           30,    "Step width to decay learning rate")
tf.app.flags.DEFINE_integer( 'MAX_EPOCH',         90,    "Maximum epoch to train")
tf.app.flags.DEFINE_float(   'eval_threshold',   0.0,    "Threshold for evaluation")

## computing parameter
tf.app.flags.DEFINE_integer(  'num_threads',                  6,   "Number of threads for input processing") #TODO: currently, use single thread for evaluation -> need to modify
tf.app.flags.DEFINE_boolean(  'use_fp16',                 False,   "Whether to use fp16")
tf.app.flags.DEFINE_integer(  'display',                     10,   "Period to display the current loss")
tf.app.flags.DEFINE_integer(  'save',                        10,   "Period to save the model (epoch)")
tf.app.flags.DEFINE_string(   'save_path',       './checkpoint',   "Path to save the model")
tf.app.flags.DEFINE_boolean(  'test_only',                False,   "Whether to test or train")
tf.app.flags.DEFINE_boolean(  'train',                     True,   "Whether to train or eval")
tf.app.flags.DEFINE_boolean(  'resume',                   False,   "Whether resume or train from scratch")
tf.app.flags.DEFINE_string(   'weights',                   None,   "Specific parameter to restore"
                                                                  "if None, restore from latest in save path")
tf.app.flags.DEFINE_integer(  'Top_k',                        1,   "Additional evaluation for top k accuracy")
tf.app.flags.DEFINE_boolean(  'Final_test',               False,   "Whether to test at the end of training")
tf.app.flags.DEFINE_integer(  'num_gpus',                     4,   "Number of gpu devices to use")


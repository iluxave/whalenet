#!/usr/bin/python3
"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('annotation_dir', '', 'Path to the directory with XML files')
flags.DEFINE_string('image_dir', '', 'Path to the directory with XML files')
flags.DEFINE_string('output', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'whale_fluke':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group):
    with tf.gfile.GFile(os.path.join(FLAGS.image_dir, group.filename), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    #an_width = group.object.width.values[0]
    #an_height = group.object.height.values[0]

    # Convert bounding boxes from original to scale/padded image
    #dim=max(an_width, an_height)
    #scale = 300.0/dim
    scale = 1

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] * scale / width)
        xmaxs.append(row['xmax'] * scale / width)
        ymins.append(row['ymin'] * scale / height)
        ymaxs.append(row['ymax'] * scale / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
#    print(classes)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    print("Example with width %d and height %d"%(width, height))
    return tf_example

# Modified From:
# https://github.comr/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import xml.etree.ElementTree as ET
import glob

def xml_to_csv(path):
    xml_list = []
    i=1
    for xml_file in glob.glob(path + '/*.xml'):
        print("Processing file %d:"%i, xml_file)
        i=i+1
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main(_):
    writer_test  = tf.python_io.TFRecordWriter("test_"+FLAGS.output)
    writer_train = tf.python_io.TFRecordWriter("train_"+FLAGS.output)
    #image_dir = os.path.join(FLAGS.image_dir)
    examples = xml_to_csv(FLAGS.annotation_dir)
    grouped = split(examples, 'filename')
    i=0
    writer=writer_test
    for group in grouped:
        i=i+1
        if i>55: writer = writer_train
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())

    writer_train.close()
    writer_test.close()

if __name__ == '__main__':
    tf.app.run()

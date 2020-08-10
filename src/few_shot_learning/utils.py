""" Utility functions. """
import numpy as np
import os
import random
import json
from typing import Dict

import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class Params():
    """Class that loads hyperparameters from a json file or a dictionary.
    Example:
    ```
    params = Params(json_path="config.json")
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params

    params = Params(config={"learning_rate":0.5})
    print(params.learning_rate)
    ```
    """

    def __init__(self, json_path: str = "", config: Dict = {}):
        if json_path:
            self.update(json_path)

        if config:
            self.dict.update(config)

    def save(self, json_path: str):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.dict, f, indent=4)

    def update(self, json_path: str):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.merge_dict(params)

    def merge_dict(self, config: Dict):
        """Merge new config with existing ones and resolve conflicts by
            - if existing config is empty overwritten by config
            - if config exits, can only update existing keys, cannot add new keys
        """
        if not self.dict:
            self.dict.update(config)
        else:
            for k, v in config.items():
                if k not in self.dict:
                    raise KeyError
                else:
                    self.dict.update({k: v})

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size

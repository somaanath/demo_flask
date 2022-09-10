import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import typing
from typing import Any,Tuple
from flask import Flask


def prediction(sen):
    reloaded = tf.saved_model.load('translator')
    result = reloaded.tf_translate(tf.constant([sen]))
    for tr in result['text']:
        return tr.numpy().decode()
 

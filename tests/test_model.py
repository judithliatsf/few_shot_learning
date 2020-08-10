from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from few_shot_learning.model import SineModel

class DataGeneratorTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dim_input = 1
        cls.dim_output = 1

    def test_CreateModel(self):
        model = SineModel()

        x = tf.constant([[1.0, 2.0]])
        self.assertAllEqual(x.shape, (1, 2))

        y = model(x)
        self.assertAllEqual(y.shape, (1, 1))  
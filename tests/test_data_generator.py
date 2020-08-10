from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from few_shot_learning.data_generator import DataGenerator
import tensorflow as tf
import numpy as np

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('datasource', 'sinusoid',
                    'sinusoid or omniglot or miniimagenet')


class DataGeneratorTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.K = 10
        cls.batch_size = 5
        cls.dim_input = 1
        cls.dim_output = 1

    def test_InitializeDataGenerator(self):        
        tf.random.set_seed(1234)
        
        # specify data class and configuration
        FLAGS.datasource = "sinusoid" 
        config = {
            'amp_range': [0.1, 5.0],
            'phase_range': [0, np.pi],
            'input_range': [-5.0, 5.0]
        }

        data_generator = DataGenerator(
            self.K, batch_size=self.batch_size, config=config)
        
        batch_x, batch_y, amp, phase = data_generator.generate(
            train=False)
        self.assertAllEqual(
            batch_x.shape, (self.batch_size, self.K, self.dim_input))
        self.assertAllEqual(
            batch_y.shape, (self.batch_size, self.K, self.dim_output))
        self.assertEqual(len(amp), self.batch_size)
        self.assertEqual(len(phase), self.batch_size)
        self.assertAllClose([amp[0], phase[0]], [1.00470823, 0.45772105])

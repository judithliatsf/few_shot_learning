import unittest
import os

from few_shot_learning.utils import Params


class ParamsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "fixtures/dummy-config.json")

    def test_merge(self):
        # load config from json
        config = Params(json_path=self.config_path)
        self.assertEqual(config.embedding_size, 1024)

        # update attribute embedding_size
        old = config.embedding_size
        config.merge_dict({"embedding_size": 512})
        self.assertNotEqual(old, config.embedding_size)

        # merge with keys not exists in Params
        with self.assertRaises(KeyError):
            config.merge_dict({"embedding": 512})

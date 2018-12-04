"""" Tests tacorn.experiment. """
import unittest
import sys
import os
import inspect
import tempfile
import shutil

CURR_DIR = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
BASE_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, BASE_DIR)

import tacorn.experiment as experiment


class TestExperiment(unittest.TestCase):
    """ Tests the experiment module. """

    def test_saveload(self):
        """ Tests saving and loading an experiment. """
        cfg = {"feature_model": "tacotron2", "wavegen_model": "wavernn"}
        tmp_dir = os.path.join(tempfile.gettempdir(),
                               "vocalid_experiment_test")
        if (os.path.exists(tmp_dir)):
            shutil.rmtree(tmp_dir)

        exp1 = experiment.create(tmp_dir, cfg)
        exp2 = experiment.load(tmp_dir)
        print(exp1)
        print(exp2)
        print(repr(exp1))
        print(repr(exp2))

        self.assertEqual(exp1, exp2)
        #self.assertTrue(len(train_df) / len(test_df) < 2.0)


if __name__ == '__main__':
    unittest.main()

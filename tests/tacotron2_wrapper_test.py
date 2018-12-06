"""" Tests tacorn.tacotron2_wrapper. """
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
import tacorn.tacotron2_wrapper as tacotron2_wrapper
import tacorn.constants as constants


class TestTacotron2Wrapper(unittest.TestCase):
    """ Tests the Tacotron2 wrapper module. """

    def test_download(self):
        """ Tests saving and loading an experiment. """
        cfg = {"feature_model": "tacotron2", "wavegen_model": "none"}
        tmp_dir = os.path.join(tempfile.gettempdir(),
                               "vocalid_taco2_download_test")
        if (os.path.exists(tmp_dir)):
            shutil.rmtree(tmp_dir)
        exp = experiment.create(tmp_dir, cfg)
        tacotron2_wrapper.download_pretrained(exp, constants.PRETRAINED_FEATURE_MODELS["lj"]["tacotron2"])
        tacotron2_wrapper._check_pretrained_model(exp)
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    unittest.main()

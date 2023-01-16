import sys
import os
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

from rAIdiologist.config.network.rAIdiologist import *
from rAIdiologist.config.network.slicewise_ran import *
from rAIdiologist.config.rAIdiologistCFG import *
from rAIdiologist.solvers import *
import unittest
import torch
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.controller import PMIController

class Test3DNetworks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test3DNetworks, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        num_slice = 30
        num_data = 4
        self.sample_input_big = torch.rand(num_data, 1, 512, 512, num_slice).cuda()
        self.sample_input = torch.rand(num_data, 1, 128, 128, num_slice).cuda()
        self.sample_input_size1 = torch.rand(1, 1, 128, 128, num_slice).cuda()
        self.sample_input_3d = torch.rand(num_data, 1, 128, 128, 128).cuda()
        self.sample_input_3d_size1 = torch.rand(1, 1, 128, 128, 128).cuda()
        self.sample_input[0, ..., 28::].fill_(0)
        self.sample_seg = torch.zeros_like(self.sample_input).cuda()
        self.sample_seg[0, 0, 50, 50, 10:20].fill_(1)
        self.sample_seg[1, 0, 50, 50, 8:15].fill_(1)
        self.sample_seg_size1 = torch.zeros_like(self.sample_input_size1).cuda()
        self.sample_seg_size1[0, 0, 50, 50, 10:20].fill_(1)
        self.expect_nonzero = torch.zeros([num_data, 1, num_slice], dtype=bool)
        self.expect_nonzero[0, ..., 9:21] = True
        self.expect_nonzero[1, ..., 7:16] = True

    def test_rAIdiologist(self):
        net = rAIdiologist(1, record=False).cuda()
        with torch.no_grad():
            for i in range(6):
                try:
                    net.set_mode(i)
                    self.assertTrue(net._mode == i)
                    out = net(self.sample_input)
                    self.assertEqual(2, out.dim(), "Failed during forward.")
                    out = net(self.sample_input_size1)
                    self.assertEqual(2, out.dim(), "Failed for batch-size = 1.")
                    print(f"Mode {i} passed.")
                except Exception as e:
                    self.fail(f"Mode {i} error. Original message {e}")


    def test_rAIdiologist_ch1(self):
        net = rAIdiologist(out_ch=1, record=False).cuda()
        with torch.no_grad():
            for i in range(6):
                try:
                    net.set_mode(i)
                    self.assertTrue(net._mode == i)
                    out = net(self.sample_input)
                    self.assertEqual(2, out.dim())
                    print(f"Mode {i} passed. Out size: {out.shape}")
                except Exception as e:
                    self.fail(f"Mode {i} error. Original message {e}")

    def test_rAIdiologist_recordon(self):
        net = rAIdiologist(out_ch=1, record=True).cuda()
        net.set_mode(5)
        with torch.no_grad():
            try:
                out = net(self.sample_input)
                self.assertNotEqual(0, len(net.get_playback()), "Playback length is zero")
                self.assertEqual(2, out.dim(), "Failed during forward.")
                out = net(self.sample_input_size1)
                self.assertEqual(2, out.dim(), "Failed for batch-size = 1.")
            except Exception as e:
                self.fail(f"Original message: {e}")
            self.assertNotEqual(0, len(net.get_playback()), "Play back is empty!")

    def test_RAN_25D(self):
        net = RAN_25D(1, 1).cuda()
        with torch.no_grad():
            out = net(self.sample_input)

            # test zero padded
            self.sample_input[0, ..., -10:] = 0
            self.sample_input[1, ..., -13:] = 0
            out = net(self.sample_input)

class TestRAIController(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._logger = MNTSLogger('.', logger_name='unittest', log_level='debug', keep_file=False, verbose=True)
        MNTSLogger.set_log_level('debug')

    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.controller = PMIController(MyControllerCFG())
        pass

    def test_s1_create(self):
        pass

    def test_s2_fit(self):
        self.controller.exec()
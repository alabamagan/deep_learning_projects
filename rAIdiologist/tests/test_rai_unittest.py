import sys
import os
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

from rAIdiologist.config.network.rAIdiologist import *
from rAIdiologist.config.network.slicewise_ran import *
from rAIdiologist.config.network.old.old_swran import SlicewiseAttentionRAN_old
from rAIdiologist.config.rAIdiologistCFG import *
from rAIdiologist.solvers import *
import unittest
import torch
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.controller import PMIController
from pytorch_model_summary.model_summary import summary

class Test3DNetworks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test3DNetworks, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        if self.__class__.__name__ == 'Test3DNetworks':
            self.skipTest('Base test class')
        num_slice = 30
        num_data = 4
        self.sample_input_big = torch.rand(num_data, 1, 512, 512, num_slice).cuda()
        self.sample_input = torch.rand(num_data, 1, 128, 128, num_slice).cuda()
        self.sample_input_size1 = torch.rand(1, 1, 128, 128, num_slice).cuda()
        self.sample_input_3d = torch.rand(num_data, 1, 128, 128, 128).cuda()
        self.sample_input_3d_size1 = torch.rand(1, 1, 128, 128, 128).cuda()
        self.sample_input[0, ..., 14::].fill_(0)
        self.sample_seg = torch.zeros_like(self.sample_input).cuda()
        self.sample_seg[0, 0, 50, 50, 10:20].fill_(1)
        self.sample_seg[1, 0, 50, 50, 8:15].fill_(1)
        self.sample_seg_size1 = torch.zeros_like(self.sample_input_size1).cuda()
        self.sample_seg_size1[0, 0, 50, 50, 10:20].fill_(1)
        self.expect_nonzero = torch.zeros([num_data, 1, num_slice], dtype=bool)
        self.expect_nonzero[0, ..., 9:21] = True
        self.expect_nonzero[1, ..., 7:16] = True
        self.EXPECTED_DIM = 2

    def expect_dim(self, out, expected_dim):
        msg = f"Dim test failed, expected dim = {expected_dim}, got shape: {out.shape}"
        self.assertEqual(expected_dim, out.dim(), msg)

    def test_input(self):
        with torch.no_grad():
            out = self.net(self.sample_input)
            print(out.shape)
            self.expect_dim(out, self.EXPECTED_DIM)

    def test_input_bsize_1(self):
        with torch.no_grad():
            out = self.net(self.sample_input_size1)
            print(out.shape)
            self.expect_dim(out, self.EXPECTED_DIM)

class TestOldSWRAN(Test3DNetworks):
    def setUp(self) -> None:
        super(TestOldSWRAN, self).setUp()
        self.net = SlicewiseAttentionRAN_old(1, 1).cuda()

class TestRAIdiologist(Test3DNetworks):
    def setUp(self) -> None:
        super(TestRAIdiologist, self).setUp()
        self.net = rAIdiologist(1, record=False).cuda()

    def test_rAIdiologist_modes(self):
        with torch.no_grad():
            for i in range(6):
                try:
                    self.net.set_mode(i)
                    self.assertTrue(self.net._mode == i)
                    self.test_input()
                    self.test_input_bsize_1()
                    print(f"Mode {i} passed.")
                except Exception as e:
                    self.fail(f"Mode {i} error. Original message {e}")

    def test_rAIdiologist_recordon(self):
        self.net.RECORD_ON = True
        self.net.set_mode(5)
        self.net = self.net.eval()
        with torch.no_grad():
            try:
                out = self.net(self.sample_input)
                self.assertNotEqual(0, len(self.net.get_playback()), "Playback length is zero")
                out = self.net(self.sample_input_size1)
            except Exception as e:
                self.fail(f"Original message: {e}")
            self.assertNotEqual(0, len(self.net.get_playback()), "Play back is empty!")

class TestRAN25D(Test3DNetworks):
    def setUp(self) -> None:
        super(TestRAN25D, self).setUp()
        self.net = RAN_25D(1, 1).cuda()

class TestRAIController(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._logger = MNTSLogger('.', logger_name='unittest', log_level='debug', keep_file=False, verbose=True)
        MNTSLogger.set_log_level('debug')

    def setUp(self) -> None:
        super(TestRAIController, self).setUp()
        self.controller.debug_mode = True
        self.controller.solver_cfg.num_of_epochs = 2
        self.controller.solver_cfg.batch_size = 1

    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        cfg = MyControllerCFG()
        cfg.data_loader_val_cfg.augmentation = '../rAIdiologist/config/v1_rAIdiologist_transform.yaml'
        self.controller = PMIController(cfg)
        pass

    def test_s1_create(self):
        pass

    def test_s2_fit(self):
        self.controller.exec()

    def test_s3_inference(self):
        self.controller.run_mode = False
        self.controller.exec()

    def test_s4_validation(self):
        with torch.no_grad():
            self.controller.debug_validation = True
            self.controller.solver_cfg._last_epoch_loss = 1E32
            self.controller.exec()
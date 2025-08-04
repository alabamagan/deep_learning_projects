import tempfile
import unittest

import torch
import torchio
from npc_segment.config.loctexthistCFG import *
from pytorch_med_imaging.solvers import SegmentationSolver
from pathlib import Path
from npc_segment.preprocess import NPCSegmentPreprocesser

class TestNPCSegment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # download dummy data
        cls.tempfolder = tempfile.TemporaryDirectory()
        cls.tempfolder_dir = Path(cls.tempfolder.name)
        cls.data = torchio.datasets.mni.Colin27()

        cls.tempfolder_dir.joinpath("img").mkdir()
        cls.tempfolder_dir.joinpath('seg').mkdir()

        # save 3 copies
        for i in range(3):
            cls.data['t1'].save(cls.tempfolder_dir / f"img/P{i}.nii.gz")
            cls.data['brain'].save(cls.tempfolder_dir / f"seg/P{i}_seg.nii.gz")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempfolder.cleanup()

    def setUp(self) -> None:
        self.cfg_controller = NPCSegmentControllerCFG()
        self.cfg_controller._data_loader_cfg.input_dir   = str(self.tempfolder_dir / "img")
        self.cfg_controller._data_loader_cfg.target_dir  = str(self.tempfolder_dir / "seg")
        self.cfg_controller._data_loader_cfg.probmap_dir = str(self.tempfolder_dir / "seg")
        self.controller = PMIController(self.cfg_controller)

    def test_solver_dataloader(self):
        c = self.controller
        # Create the solver and dataloader for they are only created within training loop
        solver: SegmentationSolver = c.solver_cls(c.solver_cfg)
        loader: PMIImageDataLoader = c.data_loader_cls(c.data_loader_cfg)

        # sampler parameters set in dataloader cfg
        target_shape = self.cfg_controller.data_loader_cfg.sampler_kwargs['patch_size']
        target_chan = 260

        # confirm type is correct
        self.assertIsInstance(loader, PMIImageDataLoader)

        # Try if dataloader is functioning properly
        for idx, row in enumerate(loader.get_torch_data_loader(10)):
            s, g = solver._unpack_minibatch(row, solver.unpack_key_forward)
            solver._logger.debug(f"{s[0].shape = }")
            solver._logger.debug(f"{s[1].shape = }")
            solver._logger.debug(f"{g.shape = }")

            self.assertEqual(s[0].shape[2:], target_shape)
            self.assertEqual(s[1].shape[1], target_chan)
            if idx > 10:
                break

    def test_net_load_checkpt(self):
        """Check if checkpoints are correct."""
        net: UNetFCAttention_p = self.controller.solver_cfg.net

        cp_path = Path("../assets/checkpoints/")
        for p in cp_path.glob("NPC_segment*.pt"):
            try:
                net.load_state_dict(torch.load(str(p.absolute())))
            except e:
                self.cfg_controller._logger.error(f"Error when loading {p}. Original error is {e}")
                raise e

    def test_net_forward(self):
        """Check if network forward is correctly implemented"""
        net = self.controller.solver_cfg.net

        # Settings
        bsize = 10
        target_shape = self.cfg_controller.data_loader_cfg.sampler_kwargs['patch_size']
        patch_size = [bsize, 1] + list(target_shape)
        feat_size = [bsize, 260]

        # create mock tensors
        mock_target = torch.ones(patch_size)
        feat = torch.ones(feat_size)

        # try to use cuda if it's available
        if torch.cuda.is_available():
            net = net.cuda()
            mock_target = mock_target.cuda()
            feat = feat.cuda()

        # pass tensor through network
        with torch.no_grad():
            out = net(mock_target, feat)

    def test_normalizer(self):
        """Check if the normalizer code is runnign correctly"""
        temp_output = tempfile.TemporaryDirectory()
        temp_output_path = Path(temp_output.name)

        # set up normalizer
        normalizer = NPCSegmentPreprocesser('../assets/normalization_t2w.yaml',
                                            state_dir='../assets/norm_states/T2WFS',
                                            training=False)
        normalizer.input_dir = self.tempfolder_dir / "img"
        normalizer.output_dir = temp_output_path

        # run
        normalizer.exec()

        # there should be a few output generated
        self.assertTrue(len(list(temp_output_path.joinpath("HuangThresholding").rglob("*nii.gz"))) > 0)

        # clean up temp directory
        temp_output.cleanup()

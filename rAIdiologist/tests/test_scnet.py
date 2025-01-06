from rAIdiologist.config.network.scnet import SCDenseNet
from rAIdiologist.config.loss.scdensenet_loss import DualLoss
import unittest
import torch
import pytorch_model_summary

class Test_SCNet(unittest.TestCase):
    def setUp(self):
        num_slice = 16
        num_data = 4
        torch.random.manual_seed(30)
        self.sample_input_big = torch.rand(num_data, 1, 384, 384, num_slice).cuda()
        self.sample_input_size1 = torch.rand(1, 1, 384, 384, num_slice).cuda()

    def test_scnet(self):
        with torch.no_grad():
            scnet = SCDenseNet().cuda()
            x_cls, x_seg = scnet(self.sample_input_big)

            # expects two outputs, one segmentation map and one benign/cancer prediction
            self.assertTupleEqual(x_seg.shape, self.sample_input_big.shape)
            self.assertTupleEqual(x_cls.shape, (x_seg.shape[0], 1))

            # try case with one sample
            x_cls, x_seg = scnet(self.sample_input_size1)
            self.assertTupleEqual(x_seg.shape, self.sample_input_size1.shape)
            self.assertTupleEqual(x_cls.shape, (1, 1))

    def test_scnet_dual_loss(self):
        scnet = SCDenseNet().cuda()
        scnet_dual_loss = DualLoss()
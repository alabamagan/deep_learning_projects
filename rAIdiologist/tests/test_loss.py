import unittest
import torch
from rAIdiologist.config.loss import *

class TestLossFunction(unittest.TestCase):
    def test_confidence_ce_loss(self):
        dummy_input = torch.rand([10, 128])
        dummy_module = torch.nn.Linear(128, 4)
        dummy_target = torch.randint(0, 1, [10])
        loss_func = ConfidenceCELoss(weight=torch.Tensor([0.5, 1.2]))

        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            dummy_module = dummy_module.cuda()
            dummy_target = dummy_target.cuda()
            loss_func = loss_func.cuda()

        o = dummy_module.forward(dummy_input)
        pred = o[:, :2]
        conf = torch.clip(torch.sigmoid(o[:, 2:]), 0.2, 0.8)

        out = loss_func((pred, conf), dummy_target)
        out.mean().backward()

        self.assertEqual(1, out.dim())

if __name__ == '__main__':
    unittest.main()

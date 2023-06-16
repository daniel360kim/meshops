from unittest import TestCase
import torch
from kinematics import apply_padding, apply_velocity

# set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



class PaddingTests(TestCase):
        
    def assertTensorsEqual(self, tensor1, tensor2):
        self.assertTrue(torch.equal(tensor1, tensor2))
    
    def setUp(self):
        self.tensor1 = torch.Tensor([
            [1, 2],
            [3, 4]
        ])
        super().setUp()
    
    def test_copy_padding(self):
        target = torch.Tensor([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ])
        self.assertTensorsEqual(target, apply_padding(self.tensor1, 2))
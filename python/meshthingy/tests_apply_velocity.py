from unittest import TestCase
import torch
from kinematics import apply_padding, apply_velocity

# set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class VelocityTests(TestCase):
        
    def assertTensorsEqual(self, tensor1, tensor2):
        for i in range(tensor1.shape[0]):
            for j in range(tensor1.shape[1]):
                self.assertAlmostEqual(tensor1[i, j].item(), tensor2[i, j].item())
    
    def setUp(self):
        self.tensor1 = torch.Tensor([
            [1.0, 0.9],
            [0.5, 0]
        ])
        super().setUp()
        
    def test_velocity_axes(self):
        target_down = torch.Tensor([
            [1.0, 0.9],
            [1.0, 0.9]
        ])
        target_right = torch.Tensor([
            [1.0, 1.0],
            [0.5, 0.5]
        ])
        target_up = torch.Tensor([
            [0.5, 0],
            [0.5, 0]
        ])
        target_left = torch.Tensor([
            [0.9, 0.9],
            [0, 0]
        ])
        
        answers = {
            0: target_right,
            90: target_up,
            180: target_left,
            270: target_down   
        }
        
        for dir in answers:
            print(f"{dir} degrees: {'-' * 50}")
            result = apply_velocity(self.tensor1, dir, device)
            self.assertTensorsEqual(result, answers[dir])
            print(f"Angle {dir} degrees passed. {'-' * 40}")
            
tester = VelocityTests()
tester.setUp()

tester.test_velocity_axes()
print("yay")
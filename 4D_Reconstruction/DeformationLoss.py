import torch
import torch.nn as nn
import torch.nn.functional as F

'''

'''

class DeformationLoss(nn.Module):
    def __init__(self, position_weight=0.5, quaternion_weight=0.5):
        super(DeformationLoss, self).__init__()
        self.position_weight = position_weight
        self.quaternion_weight = quaternion_weight

    def forward(self, predicted_x, predicted_q, target_x, target_q):

        # use MSE to measure position loss
        position_loss = F.mse_loss(predicted_x, target_x)

        # use Inner Product of predicted value and true label to measure quaterion loss
        quaternion_loss = 1 - torch.abs(torch.sum(predicted_q * target_q, dim=-1))

        # combine two loss
        loss = self.position_weight * position_loss + self.quaternion_weight * quaternion_loss.mean()

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, label_dict):
        super(DiceLoss, self).__init__()
        self.label_dict = label_dict

    def forward(self, predictions, target):        
        num_classes = len(self.label_dict)
        predictions = F.softmax(predictions, dim=1)  

        # Compute Dice loss for each class
        dice_loss = 0
        for i in range(num_classes):
            pred_class = predictions[:, i, :, :]
            target_class = torch.where(target == i+1, torch.tensor(1), torch.tensor(0))
            
            intersection = torch.sum(pred_class * target_class)
            pred_sum = torch.sum(pred_class)
            target_sum = torch.sum(target_class)
            
            dice_score = (2 * intersection) / (pred_sum + target_sum)
            dice_loss += 1 - dice_score

        return dice_loss / num_classes 

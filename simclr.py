import torch
import torch.nn.functional as F

class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        # x is a batch of feature vectors of shape (batch_size, feature_dim)
        # Compute pairwise cosine similarities between the feature vectors
        similarities = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        # Compute the "positive" similarity scores (between the augmented views of the same image)
        mask = torch.eye(x.size(0), dtype=torch.bool).to(x.device)
        positives = similarities.masked_select(mask).view(x.size(0), -1)
        # Compute the "negative" similarity scores (between the augmented views of different images)
        negatives = similarities.masked_select(~mask).view(x.size(0), -1)
        # Concatenate the positive and negative similarity scores
        logits = torch.cat([positives, negatives], dim=1)
        # Apply temperature scaling
        logits /= self.temperature
        # Compute the cross-entropy loss
        labels = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
        loss = F.cross_entropy(logits, labels)
        return loss
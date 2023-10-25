import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipScore(nn.Module):
    def __init__(self, config=None):
        super(ClipScore, self).__init__()

    def forward(self, image_features, text_features):
        with torch.no_grad():
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            # certainties, predictions = similarity.topk(1)
            anomaly_scores = similarity[:, -1]
        return anomaly_scores
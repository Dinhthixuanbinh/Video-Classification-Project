import torch.nn as nn
from transformers import VivitConfig, VivitForVideoClassification

class Model(nn.Moude):
    def __init__(self, num_classes = 2, image_size = 224, num_frames = 15):

        super(Model, self). __init__()
        cfg = VivitConfig()
        cfg.num_classes = num_classes
        cfg.image_size = image_size
        cfg.num_frames = num_frames

        self.vivit = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400",
            config = cfg,
            ignore_mismatched_sizes = True
            )
    def forward(self, x_3d):
        # (bs , C, T, H, W) -> (bs , T, C, H, W)
        x_3d = x_3d.permute(0,2,1,3,4)

        out = self.vivit(x_3d)
        return out.logits
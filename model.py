import torch
import torch.nn as nn
from torchvision import models


class UnifiedBackbone(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, num_classes=2):
        super().__init__()

        self.model_name = model_name.lower()


        RESNETS = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        }

        if self.model_name in RESNETS:
            constructor, weights_enum = RESNETS[self.model_name]
            weights = weights_enum if pretrained else None

            self.backbone = constructor(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif self.model_name in ["dinov2_small", "dinov2_base"]:
            repo = "facebookresearch/dinov2"

            if self.model_name == "dinov2_small":   # ViT-S/14
                self.backbone = torch.hub.load(repo, "dinov2_vits14", pretrained=pretrained)
                in_features = 384

            elif self.model_name == "dinov2_base": # ViT-B/14
                self.backbone = torch.hub.load(repo, "dinov2_vitb14", pretrained=pretrained)
                in_features = 768
        elif self.model_name == "retfound_green":
            self.backbone = timm.create_model('vit_small_patch14_reg4_dinov2',
            img_size=(392, 392), num_classes=0,
            checkpoint_path='C:\\Users\\preet\\Documents\\retfoundgreen_model\\retfoundgreen_statedict.pth')
            self.backbone.global_pool = 'avg'
            in_features = 384
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")


        self.classifier = nn.Linear(in_features, num_classes)


    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits



import timm 
class UnifiedBackboneMulti(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True,num_images = 2,num_classes=2):
        super().__init__()

        self.model_name = model_name.lower()

        RESNETS = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        }

        if self.model_name in RESNETS:
            constructor, weights_enum = RESNETS[self.model_name]
            weights = weights_enum if pretrained else None

            self.backbone = constructor(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif self.model_name in ["dinov2_small", "dinov2_base"]:
            repo = "facebookresearch/dinov2"

            if self.model_name == "dinov2_small":   # ViT-S/14
                self.backbone = torch.hub.load(repo, "dinov2_vits14", pretrained=pretrained)
                in_features = 384

            elif self.model_name == "dinov2_base": # ViT-B/14
                self.backbone = torch.hub.load(repo, "dinov2_vitb14", pretrained=pretrained)
                in_features = 768
        elif self.model_name == "retfound_green":
            self.backbone = timm.create_model('vit_small_patch14_reg4_dinov2',
            img_size=(392, 392), num_classes=0,
            checkpoint_path='C:\\Users\\preet\\Documents\\retfoundgreen_model\\retfoundgreen_statedict.pth')
            self.backbone.global_pool = 'avg'
            in_features = 384
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_features))


       # One encoder layer definition
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_features,
            nhead=8,
            dim_feedforward=in_features*4,
            dropout=0.1,
            batch_first=True   # important: input is [B, L, D]
        )

        # Now stack 2 layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2         # <–– exactly 2 layers
        )
        self.classifier = nn.Linear(in_features, num_classes)


    def forward(self, x):
        B, N, C, H, W = x.shape

 
        x = x.view(B * N, C, H, W)
        img_feats = self.backbone(x)           # (B*N, D)
        img_feats = img_feats.view(B, N, -1)   # (B, N, D)
    
    

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        feats_all = torch.cat(
            [cls, img_feats],
            dim=1
        )                                      # (B, 2 + M, D)
        enc = self.encoder(feats_all)
        cls_out = enc[:, 0, :] 

        logits = self.classifier(cls_out)
        return logits

import torch
import torchvision.models as models
import torch.nn as nn

# Load pretrained MaxViT-Tiny
model = models.maxvit_t(weights="DEFAULT")

print("\n=== Full Model Structure ===\n")
print(model)

print("\n=== Classifier Submodule Only ===\n")
print(model.classifier)

# Show all layers in classifier with their index
print("\n=== Classifier Layers by Index ===\n")
for idx, layer in enumerate(model.classifier):
    print(f"Index {idx}: {layer}  "
          f"{'(Linear, in_features=' + str(layer.in_features) + ')' if isinstance(layer, nn.Linear) else ''}")
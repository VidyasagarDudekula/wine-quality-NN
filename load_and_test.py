import torch

model = torch.load('wine_qt_model.pth', weights_only=False)

print(model)
model.eval()
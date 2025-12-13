#!/usr/bin/env python3
"""
Neural Style Transfer using reference images
Uses actual Britto examples to transfer style
"""

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
import copy

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_image(image_path, imsize=512):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')

    # Resize to exact size to avoid dimension mismatch
    transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    """Convert tensor to PIL image"""
    image = tensor.cpu().clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return Image.fromarray((image * 255).astype('uint8'))

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

def get_model_and_losses(cnn, content_img, style_img):
    """Build model with content and style losses"""
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    cnn = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []

    model = nn.Sequential()

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim layers after last loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:i + 1]

    return model, style_losses, content_losses

def run_style_transfer(content_img, style_img, num_steps=300, style_weight=1000000, content_weight=1):
    """Run neural style transfer"""
    print("Building model...")

    # Load VGG19
    cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

    # Get model with losses
    model, style_losses, content_losses = get_model_and_losses(cnn, content_img, style_img)

    # Start with content image
    input_img = content_img.clone()

    # Optimizer
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print(f"Running optimization ({num_steps} steps)...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}/{num_steps} - Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

if __name__ == '__main__':
    print("Loading images...")
    content = load_image('input/wilburderby1/input.png', imsize=512)
    style = load_image('input/britto_examples/Britto_Dog.png', imsize=512)

    print(f"Content shape: {content.shape}")
    print(f"Style shape: {style.shape}")

    print("\nStarting neural style transfer...")
    print("This will take 5-10 minutes on Mac Mini...")

    output = run_style_transfer(content, style, num_steps=300, style_weight=1000000, content_weight=1)

    print("\nSaving result...")
    result_img = im_convert(output)
    result_img.save('output/wilburderby1_neural_style.png')

    print("Done! Saved to: output/wilburderby1_neural_style.png")

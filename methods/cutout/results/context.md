## Research question

Modern convolutional networks have enough capacity to overfit even on standard vision benchmarks, so training usually leans on two families of regularization: data augmentation and dropout-like noise. Image augmentation is cheap and effective, and the common transforms are mostly geometric or photometric. Dropout is simple and broadly used, applied in fully connected layers and, in various forms, inside convolutional layers.

The question is how to regularize a convolutional classifier so that it improves on an already strong baseline that uses batch normalization and ordinary augmentation, while keeping the classifier architecture, loss, and test-time pipeline unchanged.

## Background

The standard small-image recognition setup is already crowded with helpful machinery: residual or wide-residual networks, batch normalization, SGD with momentum, weight decay, random crops, horizontal flips, and per-channel normalization. A useful new regularizer is compared against this strong baseline.

Dropout was designed to discourage co-adaptation of units. In convolutional layers, two properties differ from the dense case. First, convolutional layers have far fewer parameters than dense layers. Second, nearby image pixels and nearby feature activations are highly correlated, so pointwise removal of an activation behaves more like noise injection than like the exponential model averaging dropout produces in dense layers.

Several feature-map-level dropout variants have been proposed for this spatially correlated setting; they operate on the feature maps inside the network.

## Baselines

**Dropout (Hinton et al., 2012; Srivastava et al., 2014).** During training, hidden units are randomly set to zero; at test time a corresponding scaling rule approximates averaging many thinned networks. It discourages co-adaptation and is widely used in dense layers.

**SpatialDropout (Tompson et al., 2015).** Entire feature maps are dropped rather than individual spatial positions, which removes within-map neighbor redundancy in the dropped channel.

**Max-drop and stochastic dropout variants (Park & Kwak, 2016; Wu & Gu, 2015).** These methods alter which convolutional activations or pooling outputs are removed, sometimes focusing on strong activations or changing the drop probability. They show that the placement and structure of dropout matter in CNNs.

**Denoising autoencoders (Vincent et al., 2010).** Inputs are corrupted and the model is trained to reconstruct the original, learning representations from damaged inputs under a reconstruction objective.

**Context encoders (Pathak et al., 2016).** A larger missing image region is reconstructed from its surroundings, using an encoder-decoder reconstruction task that forces more global semantic understanding than scattered pixel corruption.

**Occlusion-style augmentation.** Earlier image augmentation work includes fake scratches, dots, scribbles, and partial occlusions on characters, motivated by the goal that recognition systems should not depend on a single always-visible local cue.

## Evaluation settings

The core benchmarks are CIFAR-10 and CIFAR-100: 50,000 training and 10,000 test RGB images at 32x32, with 10 and 100 classes. SVHN adds a digit-recognition setting with 73,257 official training examples, 531,131 extra training examples, and 26,032 test examples at the same resolution. STL-10 is a low-label, higher-resolution probe: 5,000 labeled training images and 8,000 test images at 96x96.

The relevant training baselines use ResNet-18, WideResNet-28-10, WideResNet-16-8, and shake-shake ResNet/ResNeXt models. CIFAR training uses per-channel normalization, optional zero-padding by 4 pixels followed by a random 32x32 crop, horizontal mirroring with probability 0.5, batch size 128, SGD with Nesterov momentum 0.9, weight decay 5e-4, and step-decayed learning rates. Hyperparameters for any new regularizer are selected on a held-out 10% validation split before full training.

The main metric is test error rate. The comparison also checks whether the regularizer remains useful on models that already include batch normalization, convolutional dropout, and standard augmentation.

## Code framework

The available implementation surface is a normal torchvision data pipeline feeding an unchanged classifier, cross-entropy loss, SGD optimizer, and learning-rate scheduler.

```python
import torch
from torchvision import datasets, transforms


if dataset == "svhn":
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
        std=[x / 255.0 for x in [50.1, 50.6, 50.8]],
    )
else:
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

train_transform = transforms.Compose([])
if data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
# TODO: insert the regularization/augmentation step here, if the method needs one.

train_dataset = datasets.CIFAR10(
    root="data/",
    train=True,
    transform=train_transform,
    download=True,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
)

# model, nn.CrossEntropyLoss(), SGD(Nesterov, momentum=0.9, weight_decay=5e-4),
# and MultiStepLR are otherwise unchanged.
```

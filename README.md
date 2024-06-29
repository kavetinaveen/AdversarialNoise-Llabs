# ML Programming Problem - Leap Labs

## Problem Statement:

Introduce adversarial noise into an input image to trick an image classification model into misclassifying it as the desired target class.

## Approach:

There are multiple ways to introduce the adversarial noise into an image. I am implementing three gradient based approaches:

- Fast Gradients Sign Method (FGSM)
- Basic Iterative Method (BIM)
- Projected Gradient Descent (PGD)

### Fast Gradients Sign Method (FGSM):

It adds or subtracts $\epsilon$ from each pixel of the image in the direction of the gradient of the loss.

**Hyper parameters:** $\epsilon$

### Basic Iterative Method (BIM):

It iterates FGSM `n` number of times, at each iteration adds $\alpha$ to each pixel of the image and also ensures the perturbations at any given iteration are not beyond the $\epsilon$ neighbourhood of the input image. Intuitively, FGSM takes one big step whereas BIM takes `n` constrained small steps.

**Hyper parameters:** $\epsilon$, $\alpha$ and `n`

### Projected Gradient Descent (PGD):

PGD is also iterative process, same as BIM. At each iteration, it perturbs the image first and then projects perturbed image back into $\epsilon$ neighbourhood of the input image.

**Hyper parameters:** $\epsilon$, $\alpha$ and `n`

## Model:

Using Resnet50 pre-trained model from `torchvision`


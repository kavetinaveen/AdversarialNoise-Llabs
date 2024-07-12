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

## How to Run?

Packaged the code and uploaded into [pypi](https://pypi.org/project/AdNoise/0.0.4/)

**Option 1 (Recommended):**
- Install the library from pypi `pip install AdNoise`
- Clone this repository
- Run the [example notebook](https://github.com/kavetinaveen/AdversarialNoise-Llabs/blob/main/example.ipynb)

The current version only supports the local image paths. If you want to try a new image, save it in your local and update `image_path` and `label` parameters accordingly in the notebook.

**Option 2:**
- Clone this repository
- Change your working directory to this repo in your local
- Enable or disable an attack type by adding or removing from the noise type in the config
- Run `python AdversarialNoise.py -c "./config.yaml" -i ./../data/input/n01491361_tiger_shark.jpeg -l "robin"`
- The perturbed images should be saved at `data/output/<input_image_file_name>_<type_of_attack>.jpeg`

If you're using option 2 please make sure your versions match with the below:

- python: `3.11.4`
- torch: `2.3.1`
- torchvision: `0.18.1`


import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm

class AttackMethods:
    def __init__(self):
        pass

    def fgsm_attack(self, image, grad_data, epsilon):
        assert isinstance(grad_data, torch.Tensor), 'grad_data must be a torch.Tensor'
        # Computing perturbations in the direction of the gradient of the loss
        perturbation = epsilon * grad_data.sign()
        # Adding the perturbations to the input image
        image = image - perturbation
        return image
    
    def bim_attack(self, model, image, target_class, epsilon, alpha, num_iter):
        assert isinstance(model, torch.nn.Module), 'model must be a torch.nn.Module'
        assert isinstance(image, torch.Tensor), 'image must be a torch.Tensor'
        # Ensuring both model and image are on the same device
        image.to(next(model.parameters()).device)
        # Enabling required_grad to access the gradients of the input image
        perturbed_image = image.clone().detach().requires_grad_(True)
        # Converting the integer target label to tensor
        target_class = torch.tensor([target_class])
        
        for _ in tqdm(range(num_iter)):
            # Generating the model predictions for the perturbed image
            predictions = model(perturbed_image)
            ce_loss = cross_entropy(predictions, target_class)
            # Setting all other (other than the input image) gradients to zero
            model.zero_grad()
            # Computing the gradient of the loss for the input image
            ce_loss.backward()
            
            with torch.no_grad():
                # Perform FGSM
                grad_data = perturbed_image.grad.data
                perturbed_image = self.fgsm_attack(perturbed_image, grad_data, alpha)
                # Clip the perturbations to epsilon bounds
                perturbations = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
                perturbed_image = torch.clamp(image + perturbations, min=0, max=1)
            # Set the required_grad to True for the perturbed image to access the non-leaf gradients
            perturbed_image.requires_grad = True    
            perturbed_image.retain_grad()
            if perturbed_image.grad is not None:
                perturbed_image.grad.zero_()
        return perturbed_image
    
    def pgd_attack(self, model, image, target_class, epsilon, alpha, num_iter):
        assert isinstance(model, torch.nn.Module), 'model must be a torch.nn.Module'
        assert isinstance(image, torch.Tensor), 'image must be a torch.Tensor'
        # Ensuring both model and image are on the same device
        image.to(next(model.parameters()).device)
        # Enabling required_grad to access the gradients of the input image
        perturbed_image = image.clone().detach().requires_grad_(True)
        # Converting the integer target label to tensor
        target_class = torch.tensor([target_class])
        
        for _ in tqdm(range(num_iter)):
            # Generating the model predictions for the perturbed image
            predictions = model(perturbed_image)
            ce_loss = cross_entropy(predictions, target_class)
            # Setting all other (other than the input image) gradients to zero
            model.zero_grad()
            # Computing the gradient of the loss for the input image
            ce_loss.backward()
            
            with torch.no_grad():
                # Perform the update
                grad_data = perturbed_image.grad.data
                perturbed_image = self.fgsm_attack(perturbed_image, grad_data, alpha)
                # Project the perturbed image into the allowable epsilon bounds
                perturbed_image = torch.clamp(perturbed_image, min=image-epsilon, max=image+epsilon)
                perturbed_image = torch.clamp(perturbed_image, min=0, max=1)
            # Set the required_grad to True for the perturbed image to access the non-leaf gradients
            perturbed_image.requires_grad = True
            perturbed_image.retain_grad()
            if perturbed_image.grad is not None:
                perturbed_image.grad.zero_()
        return perturbed_image
        
        
        
import numpy as np
import pandas as pd
import yaml
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import cross_entropy
from torchvision import models
import torchvision.transforms as transforms
import argparse

import logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class AdversarialNoise:
    def __init__(self, params, image_path, label):
        self.model = getattr(models, params['model']['model_name'])(weights=True)
        self.model.eval()
        
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])
        
        self.labels_df = pd.read_csv(params['directories']['labels'])
        self.noise_type = params['noise']['type']
        self.epsilon = params['noise']['FGSM']['epsilon']
        self.output_dir = params['directories']['output']
        self.image_path = image_path
        self.label = label
        self.target = self.labels_df[self.labels_df["label_text"].str.contains(self.label)]["label_digit"].values[0]
        
    def read_image(self):
        try:
            image = Image.open(self.image_path)
            original_size = image.size
            logging.info('Image found, displaying input image')
            # plt.imshow(image)
            # plt.show()
        except:
            raise Exception('Image not found, please check the path')
        image = image.convert('RGB')
        transformed_image = self.transform(image)
        transformed_image = transformed_image.unsqueeze(0)
        return transformed_image, original_size
    
    def compute_grad(self, image, target_class):
        assert isinstance(image, torch.Tensor), 'image must be a torch.Tensor'
        assert isinstance(target_class, (int, np.int64)), 'target_class must be an integer'
        assert isinstance(self.labels_df, pd.DataFrame), 'labels_df must be a pandas DataFrame'
        assert target_class in self.labels_df["label_digit"].values, 'target_class not found in labels_df'
        
        target_class = torch.tensor([target_class])
        image.requires_grad = True
        
        predictions = self.model(image)
        ce_loss = cross_entropy(predictions, target_class)
        logging.info(f"Cross Entropy Loss: {ce_loss.item()}")
        
        _, idx = torch.max(predictions, 1)
        prediction_digit = self.labels_df[self.labels_df["label_digit"] == idx.item()]["label_digit"].values[0]
        prediction_text = self.labels_df[self.labels_df["label_digit"] == idx.item()]["label_text"].values[0]
        
        target_text = self.labels_df[self.labels_df['label_digit'] == target_class.item()]['label_text'].values[0]
        logging.info(f"Actual: {target_class.item()} - {target_text}")
        logging.info(f"Prediction: {prediction_digit} - {prediction_text}")
        
        if prediction_digit != target_class.item():
            logging.info("Model prediction is incorrect, no need of gradient computation")
            return None
        
        self.model.zero_grad()
        ce_loss.backward()
        return image.grad.data
    
    def fgsm_attack(self, image, grad_data):
        assert isinstance(grad_data, torch.Tensor), 'grad_data must be a torch.Tensor'
        perturbation = self.epsilon * grad_data.sign()
        image = image + perturbation
        return image
    
    def predict_perturbed_image(self, perturbed_image, target_class):
        predictions = self.model(perturbed_image)
        ce_loss = cross_entropy(predictions, torch.tensor([target_class]))
        logging.info(f"Cross Entropy Loss after perturbation: {ce_loss.item()}")
        
        _, idx = torch.max(predictions, 1)
        prediction_digit = self.labels_df[self.labels_df["label_digit"] == idx.item()]["label_digit"].values[0]
        prediction_text = self.labels_df[self.labels_df["label_digit"] == idx.item()]["label_text"].values[0]
        logging.info(f"Prediction after perturbation: {prediction_digit} - {prediction_text}")
    
    def save_image(self, image, original_size, noise_type):
        assert isinstance(image, torch.Tensor), 'image must be a torch.Tensor'
        save_path = f"{self.output_dir}{self.image_path.split('/')[-1].split('.')[0]}_{noise_type}.jpeg"
        image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.detach().numpy()
        image = transforms.ToPILImage()(image)
        image = image.resize(original_size, Image.LANCZOS)
        try:
            image.save(save_path)
        except:
            raise Exception('Error saving image, please check the path')
        # plt.imshow(image)
        # plt.show()
        
    def run(self):
        logging.info('Reading image')
        image, original_size = self.read_image()
        logging.info('Computing gradient')
        grad_data = self.compute_grad(image, self.target)
        if grad_data is not None:
            if "FGSM" in self.noise_type:
                logging.info('Perturbing image with FGSM attack')
                logging.info('====================================================')
                perturbed_image = self.fgsm_attack(image, grad_data)
                self.predict_perturbed_image(perturbed_image, self.target)
                logging.info('Saving perturbed image')
                self.save_image(perturbed_image, original_size, "FGSM")
            else:
                logging.info('Model prediction is incorrect, no need of perturbation')
                perturbed_image = image
        
if __name__ == '__main__':
    params = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--image_path', required=True, help='Provide path to image')
    arg_parser.add_argument('-l', '--label', required=True, help='Provide target label')
    
    args = vars(arg_parser.parse_args())
    image_path = args["image_path"]
    label = args["label"]
    noise = AdversarialNoise(params, image_path, label)
    noise.run()
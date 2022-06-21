from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os


def load_data(input_size, batch_size, data_dir):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(0.2),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                                   hue=0.5),
            transforms.RandomSolarize(threshold=128),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(0.2),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                                   hue=0.5),
            transforms.RandomSolarize(threshold=128),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                      num_workers=4) for x in ['train', 'test']}

    return dataloaders_dict

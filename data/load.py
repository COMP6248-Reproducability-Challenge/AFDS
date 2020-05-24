import os

from torchvision import transforms, datasets
from data.stanford_dogs_data import dogs


# def transform_compose(image_size):
#     crop_size = {299: 320, 224: 256}
#     resize = crop_size[image_size]
#     hflip = transforms.RandomHorizontalFlip()
#     rcrop = transforms.RandomCrop((image_size, image_size))
#     ccrop = transforms.CenterCrop((image_size, image_size))
#     totensor = transforms.ToTensor()
#     cnorm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean and std for imagenet
#     r = [transforms.Resize(resize), hflip, ccrop, rcrop, totensor, cnorm]
#     return transforms.Compose(r)


def load_datasets(set_name, input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    if set_name == 'stanford_dogs':
        #input_transforms = transform_compose(input_size)

        train_dataset = dogs(root='./data',
                             train=True,
                             cropped=False,
                             transform=data_transforms['train'],
                             download=True)
        test_dataset = dogs(root='./data',
                            train=False,
                            cropped=False,
                            transform=data_transforms['test'],
                            download=True)

        classes = train_dataset.classes

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()
    else:
        return None, None

    return train_dataset, test_dataset, classes

# %%

import torch
import os
from torch.utils.data import DataLoader, TensorDataset



# %%
def corrupt_mnist():

    # need to define it inside so that when called it can find the path
    path = "/home/krrish/home/desktop/astha_programming/comp-viz/ml_operations/corruptmnist_v1/"

    # train data concatation
    train_data_list = []
    train_data_list_labels = []
    for i in range(5):
        train_data = torch.load(os.path.join(path, f"train_images_{i}.pt"))
        train_data_list.append(train_data)

        train_data_labels = torch.load(os.path.join(path, f"train_target_{i}.pt"))
        train_data_list_labels.append(train_data_labels)

    train_images = torch.cat(train_data_list, dim=0)
    train_data_labels = torch.cat(train_data_list_labels, dim=0)

    # test dat aconcatation
    test_images = torch.load(os.path.join(path, "test_images.pt"))
    test_images_labels = torch.load(os.path.join(path, "test_target.pt"))

    # mapping the labels
    train_images = train_images.reshape(-1,1,28,28)
    test_images = test_images.reshape(-1,1,28,28)

    # combining the images with labels
    train_dataset = TensorDataset(train_images, train_data_labels)
    test_dataset = TensorDataset(test_images, test_images_labels)

    # data interators
    train_dataloaders = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloaders = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataloaders , test_dataloaders
    

# %%
# checking the shape of the image 
train, test_dataloaders =corrupt_mnist()
image, label = next(iter(test_dataloaders))
image.shape



# %%

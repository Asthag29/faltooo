# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import corrupt_mnist
# from main import train_validation
from model import Network
import torch

# %%
def classification(batch_no,image_path):
    
    model=Network() #making instance again because path only stores the parameter not the complete model itself

    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    _, testdata = corrupt_mnist()
    
    # Check if batch_no is valid
    # if batch_no >= len(testdata):
    #     print(f"Error: batch_no {batch_no} is out of range. Total batches: {len(testdata)}")
    #     return #adding return so that the function breaks here
    
    # Iterate through batches to reach the desired batch_no
    for current_batch, (images, labels) in enumerate(testdata):
        if current_batch == batch_no:
            
            out= model(images)
            
            _, predictions = torch.max(out, dim=1)
            for i in range(len(predictions)):
                if predictions[i] != labels[i]:
            
                    # Extract the specific image and label
                    image = images[i].squeeze().numpy() #accessing the element corresponding to the index then squueze() by defualt remove all the singleton element
                    
                    label = labels[i].item()
                    
                    # Plot the image
                    plt.figure(figsize=(6, 6))
                    plt.imshow(image, cmap='gray')
                    plt.title(f'Batch: {batch_no}, Index: {i}, Label: {label}, Predicted_label: {predictions[i]}')
                    plt.axis('off')
                    plt.show()
                    plt.savefig(image_path)


                    print(f"Batch number: {batch_no}")
                    print(f"Index in batch: {i}")
                    print(f"Label: {label}")
                    print(f"predicted: {predictions[i]}")
                    break





# %%

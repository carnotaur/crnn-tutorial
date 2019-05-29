import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import clear_output
from .utils import decode_prediction



def plot_loss(epoch: int, 
              train_losses: list, 
              val_losses: list, 
              n_steps: int = 100):
    """
    Plots train and validation losses 
    """
    # clear previous graph
    clear_output(True)
    # making titles
    train_title = f'Epoch:{epoch} | Train Loss:{np.mean(train_losses[-n_steps:]):.6f}'
    val_title = f'Epoch:{epoch} | Val Loss:{np.mean(val_losses[-n_steps:]):.6f}'

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_losses)
    ax[1].plot(val_losses)

    ax[0].set_title(train_title)
    ax[1].set_title(val_title)

    plt.show()

def print_prediction(model, dataset, device, label_converter):
    idx = np.random.randint(len(dataset))
    path = dataset.pathes[idx]
    
    with torch.no_grad():
        model.eval()
        img, target_text = dataset[idx]
        img = img.unsqueeze(0)
        logits = model(img.to(device))
        
    pred_text = decode_prediction(logits.cpu(), label_converter)

    img = np.asarray(Image.open(path).convert('L'))
    title = f'Truth: {target_text} | Pred: {pred_text}'
    plt.imshow(img)
    plt.title(title)
    plt.axis('off');
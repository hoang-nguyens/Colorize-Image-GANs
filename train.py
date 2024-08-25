from data import *
import glob
from tqdm.notebook import  tqdm
from utils import *
from models import *
dir = "Directory to Folder"
dir = r"D:\Colorization\food"

np.random.seed(42)

image_paths = np.array(glob.glob(dir + '/*.jpg')) # collect all image files to numpy array

paths_len = len(image_paths)
train_len = int(paths_len * 0.8)

random_idxs = np.random.permutation(paths_len)
train_idxs = random_idxs[:train_len]
val_idxs = random_idxs[train_len:]

train_paths = image_paths[train_idxs]
val_paths = image_paths[val_idxs]

train_dataloader = ColorizationDataLoader(paths = train_paths, mode = 'train')
val_dataloader = ColorizationDataLoader(paths = val_paths, mode = 'val')

def train_model(model, train_dataloader, epochs, display_every=200, resume_training = False):
    data = next(iter(val_dataloader)) # getting a batch for visualizing the model output after fixed intrvals
    optimizers = [model.model_G_optim, model.model_D_optim]
    if resume_training:
        model, optimizers, start_epoch, _ = load_checkpoint('checkpoint.pth', model, optimizers)


    for epoch in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dataloader):
            model.setup_input(data)
            model.optimize()

            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dataloader)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=True) # function displaying the model's output
                losses = {
                    'loss_D': model.loss_D,
                    'loss_G_GAN': model.loss_G_GAN,
                    'loss_G_L1': model.loss_G_L1,
                    'loss_G': model.loss_G
                }

                save_checkpoint(model, optimizers, epoch, losses, 'checkpoint.pth')

if __name__ == '__main__':
    model = GANs()
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Move model to GPU if available
    train_model(model, train_dataloader, 10)
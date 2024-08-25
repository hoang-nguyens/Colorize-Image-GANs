import PIL
import torch
from models import GANs
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb

if __name__ == '__main__':
    model = GANs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    path = r"D:\Colorization\food\300.jpg"
    img = PIL.Image.open(path).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img)

    lab_img = rgb2lab(img).astype(np.float32)

    L_channel = lab_img[:, :, 0] / 50.0 - 1.0

    L_channel_tensor = torch.from_numpy(L_channel).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model.net_G(L_channel_tensor.to(device).float())

    preds = preds.squeeze(0).cpu().numpy()
    preds = (preds * 128).clip(-128, 127)
    L_channel = (L_channel_tensor.squeeze(0).squeeze(0).cpu().numpy() + 1.0) * 50.0

    lab_result = np.concatenate([L_channel[:, :, np.newaxis], preds.transpose(1, 2, 0)], axis=-1)

    colorized = lab2rgb(lab_result)

    plt.imshow(np.clip(colorized, a_min=0.0, a_max=1.0))
    plt.axis('off')
    plt.show()

    print(f"Preds min: {preds.min()}, Preds max: {preds.max()}")
    total_params = sum(p.numel() for p in model.parameters())

import zipfile

# Specify the file name
model_file = 'checkpoint.pth'
compressed_file = 'model.zip'

# Compress the .pth file
with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(model_file)


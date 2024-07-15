import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.quantization import quantize_dynamic, get_default_qconfig
import torch.nn.utils.prune as prune

# Constants and parameters
data_dir = 'D:\\Code'
pixelated_dir = 'D:\\Code\\pixelated'
output_dir = 'D:\\Code\\output'
img_extensions = ['jpeg', 'jpg', 'bmp', 'png']
min_resolution_width = 1920
min_resolution_height = 1080
min_width, min_height = 256, 256
batch_size = 4
accumulation_steps = 16

# Function to pixelate images
def pixelate_images(folder_path, scale_factor, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pixelated_images = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # Skip non-image files

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}, skipping pixelation.")
            continue

        try:
            # Downscale the image
            small = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            # Upscale the image back to the original size
            pixelated = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            pixelated_images.append(pixelated)

            # Construct the output file path
            output_path = os.path.join(output_folder, f"pixelated_{filename}")

            # Save the pixelated image
            if cv2.imwrite(output_path, pixelated):
                print(f"Pixelated image saved to {output_path}")
            else:
                print(f"Failed to save image: {output_path}")
        except Exception as e:
            print(f"Error processing image: {image_path} - {e}")

    return pixelated_images

# Paths and parameters for pixelation
pr_train_folder = 'D:\\Code\\pr_train'
scale_factor = 0.4
pixelated_output_folder = 'D:\\Code\\pixelated'

# Pixelate images in pr_train folder
pixelate_images(pr_train_folder, scale_factor, pixelated_output_folder)

# Function to preprocess image
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        height, width, _ = image.shape
        
        if width < min_resolution_width or height < min_resolution_height:
            print(f"Skipping image {image_path}: resolution {width}x{height} is less than {min_resolution_width}x{min_resolution_height}")
            return None

        # Resize original image
        image = cv2.resize(image, (min_width, min_height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0

        return image.astype(np.float32)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Custom dataset class
class ImageDataset(Dataset):
    def _init_(self, folder_path, transform=None):
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.transform = transform

    def _len_(self):
        return len(self.image_files)

    def _getitem_(self, idx):
        image_path = self.image_files[idx]
        image = preprocess_image(image_path)
        if image is not None:
            if self.transform:
                image = self.transform(image)
            return image
        return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        print("Warning: Batch is empty after filtering None values.")
        return None  # Handle this case appropriately
    
    try:
        return torch.stack(batch)
    except Exception as e:
        print(f"Error stacking tensors: {e}")
        return None  # Handle this case appropriately

# Function to build a custom model with VGG16
class CustomVGG16Model(nn.Module):
    def _init_(self):
        super(CustomVGG16Model, self)._init_()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features[:23]  # Use fewer layers from VGG16
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.0)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.gaussian_blur(x)
        x = self.decoder(x)
        return x

# Gaussian blur module
class GaussianBlur(nn.Module):
    def _init_(self, kernel_size, sigma):
        super(GaussianBlur, self)._init_()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create a Gaussian kernel
        x = torch.arange(kernel_size).float()
        x -= (kernel_size - 1) / 2
        gauss = torch.exp(-(x * 2) / (2 * sigma * 2))
        gauss = gauss / gauss.sum()
        self.gaussian_kernel = gauss[:, None] * gauss[None, :]

    def forward(self, x):
        # Apply Gaussian blur
        channels = x.shape[1]
        kernel = self.gaussian_kernel.expand(channels, 1, -1, -1).to(x.device)
        padding = self.kernel_size // 2
        x = nn.functional.conv2d(x, kernel, padding=padding, groups=channels)
        return x

# Main function for training and evaluation
def main():
    # Paths to training and validation image folders
    train_folder_path = pixelated_dir
    val_folder_path = pixelated_dir
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = ImageDataset(train_folder_path, transform=transform)
    val_dataset = ImageDataset(val_folder_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # Build and compile the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomVGG16Model().to(device)
    model.qconfig = get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop with gradient accumulation
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, images in enumerate(train_loader):
            if images is None:
                continue  # Skip None batches

            images = images.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if outputs.size() != images.size():
                    outputs = nn.functional.interpolate(outputs, size=images.size()[2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, images)
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Apply quantization
    torch.quantization.convert(model, inplace=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, 'satvik_rohan.pth'))
    print("Model training and saving completed successfully.")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images in val_loader:
            if images is None:
                continue  # Skip None batches
            
            images = images.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                outputs = nn.functional.interpolate(outputs, size=images.size()[2:], mode='bilinear', align_corners=False)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(images.cpu().numpy())

    # Convert lists to numpy arrays and flatten them
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    precision = precision_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
    recall = recall_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='binary')

    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation Precision: {precision:.4f}')
    print(f'Validation Recall: {recall:.4f}')
    print(f'Validation F1 Score: {f1:.4f}')

# Ensure the script runs when executed directly
if __name__ == '__main__':
    main()

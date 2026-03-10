# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## THEORY


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1: Load and Preprocess Dataset
Load image dataset, normalize pixel values to [0, 1], and prepare training and testing sets. Create DataLoaders for batch processing.

### STEP 2: Add Noise to Images
Introduce Gaussian noise to clean images to create noisy versions for training the denoising autoencoder.

### STEP 3: Design Convolutional Autoencoder Architecture
Build an encoder with convolutional layers to compress noisy images into a latent representation, and a decoder with transposed convolutional layers to reconstruct clean images.

### STEP 4: Compile and Configure the Model
Define Mean Squared Error (MSE) as the loss function, Adam optimizer for weight updates, and set hyperparameters like learning rate and batch size.

### STEP 5: Train the Autoencoder
Train the model on noisy images to reconstruct clean images. Monitor training loss over epochs and validate on test set to ensure convergence.

### STEP 6: Evaluate and Visualize Results
Test the autoencoder on noisy images, compare original, noisy, and reconstructed images. Calculate reconstruction quality metrics (MSE, PSNR) and visualize denoising performance.


## PROGRAM

### Name: KARTHIK B

### Register Number: 212224230118

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1), # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, output_padding=1, padding=1),  # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

# Initialize model
```python
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```
# Training function
```python
# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

```

# Visualization function
```python
# Evaluate and visualize
def visualize_denoising(model, loader, start=15,end=25):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    num_images=end-start
    plt.figure(figsize=(18, 6))
    for i in range(start,end):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

```

### OUTPUT

### Model Summary
![alt text](image.png)

### Training loss

![alt text](image-1.png)
## Original vs Noisy Vs Reconstructed Image
![alt text](image-2.png)

## RESULT

The convolutional autoencoder was successfully trained for image denoising.

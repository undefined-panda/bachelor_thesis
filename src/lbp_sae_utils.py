import matplotlib.pyplot as plt

def train_autoencoder(model, optimizer, dataloader, num_epochs=20):
    for epoch in range(num_epochs):
        for img, _ in dataloader:
            optimizer.zero_grad()
            # Vorwärtsdurchlauf
            output = model(img)
            loss = model.loss_function(output, img)
        
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def plot_lbp_images(lbp_images, typical_images):
    for key in lbp_images.keys():
        selected_images = typical_images[key]
        new_images = lbp_images[key]
        fig, axes = plt.subplots(2, len(selected_images), figsize=(15, 5))
        for i in range(len(selected_images)):
            axes[0, i].imshow(selected_images[i][0], cmap='gray')
            axes[0, i].axis('off')

            axes[1, i].imshow(new_images[i][0], cmap='gray')
            axes[1, i].axis('off') 
        
        plt.show()
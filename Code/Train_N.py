##########################################################################
######################PRETRAINED RESNET AND LSTM########################
##########################################################################

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.models import resnet50, ResNet50_Weights
# from torchvision import transforms  # Added missing import
# from data_loader import get_data_loader
# from model import ImageEncoder, CaptionDecoder

# # Hyperparameters
# num_epochs = 2  # Define the number of epochs for training
# batch_size = 16
# learning_rate = 0.001
# embedding_dim = 256
# hidden_dim = 512

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Paths to COCO dataset
# images_dir = '../COCO_Data/train2017'
# captions_path = '../COCO_Data/annotations/captions_train2017.json'

# # Load data
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# dataloader = get_data_loader(images_dir, captions_path, vocab_exists=False, batch_size=batch_size, transform=transform)

# # Initialize models
# encoder = ImageEncoder(embedding_dim=embedding_dim).to(device)
# decoder = CaptionDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(dataloader.dataset.vocab)).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# # Training loop
# print("Starting training...")
# for epoch in range(num_epochs):
#     encoder.train()
#     decoder.train()

#     for i, batch in enumerate(dataloader):
#         images = batch['images'].to(device)
#         captions = batch['tokenized_caption'].to(device)

#         # Forward pass
#         features = encoder(images)
#         outputs = decoder(features, captions)
        
#         # Compute loss
#         loss = criterion(outputs.view(-1, len(dataloader.dataset.vocab)), captions.view(-1))

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Print progress
#         if i % 10 == 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

#     # Save the model after each epoch
#     torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
#     torch.save(decoder.state_dict(), f"decoder_epoch_{epoch+1}.pth")
#     print(f"Epoch {epoch+1}/{num_epochs} completed. Models saved.")

# print("Training completed successfully!")

##########################################################################
###################### CNN AND RNN ########################
##########################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms  # Added missing import
from data_loader import get_data_loader
# from model import ImageEncoder, CaptionDecoder
from model import CustomCNNEncoder, CustomRNNDecoder

# # Hyperparameters
num_epochs = 2  # Define the number of epochs for training
batch_size = 16
# learning_rate = 0.001
embedding_dim = 256
hidden_dim = 512

# Initialize custom models
# embedding_dim = 256
# hidden_dim = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to COCO dataset
images_dir = '../COCO_Data/train2017'
captions_path = '../COCO_Data/annotations/captions_train2017.json'

# Load data
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataloader = get_data_loader(images_dir, captions_path, vocab_exists=False, batch_size=batch_size, transform=transform)

# # Initialize models
# encoder = ImageEncoder(embedding_dim=embedding_dim).to(device)
# decoder = CaptionDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(dataloader.dataset.vocab)).to(device)

encoder = CustomCNNEncoder(embedding_dim=embedding_dim).to(device)
decoder = CustomRNNDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(dataloader.dataset.vocab)).to(device)

# Set device
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()

    for i, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        captions = batch['tokenized_caption'].to(device)

        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Compute loss
        loss = criterion(outputs.view(-1, len(dataloader.dataset.vocab)), captions.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Save the model after each epoch
    torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
    torch.save(decoder.state_dict(), f"decoder_epoch_{epoch+1}.pth")
    print(f"Epoch {epoch+1}/{num_epochs} completed. Models saved.")

print("Training completed successfully!")

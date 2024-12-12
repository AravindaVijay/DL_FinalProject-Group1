##########################################################################
###################### CUSTOM CNN RNN ###################################
##########################################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data_loader import get_data_loader  

class CustomCNNEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CustomCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, images):
        x = torch.relu(self.conv1(images))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x

#Custom RNN Decoder
class CustomRNNDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(CustomRNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, captions):
        embeddings = self.embedding(captions[:, :-1])  
        inputs = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
        rnn_output, _ = self.rnn(inputs)
        outputs = self.fc(rnn_output)
        return outputs

    def generate_caption(self, inputs, states=None, max_length=20):
        batch_size = inputs.size(0)
        if states is None:
            states = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(inputs.device)

        generated_ids = []
        inputs = inputs.unsqueeze(1)
        for _ in range(max_length):
            rnn_output, states = self.rnn(inputs, states)
            outputs = self.fc(rnn_output.squeeze(1))
            predicted_token = outputs.argmax(dim=1)
            generated_ids.append(predicted_token.item())
            if predicted_token == 1:  
                break
            inputs = self.embedding(predicted_token).unsqueeze(1)
        return generated_ids

num_epochs = 2
batch_size = 16
learning_rate = 0.001
embedding_dim = 256
hidden_dim = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the dataset
images_dir = 'path_to_images'
captions_path = 'path_to_annotations'

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataloader = get_data_loader(images_dir, captions_path, vocab_exists=False, batch_size=batch_size, transform=transform)

encoder = CustomCNNEncoder(embedding_dim=embedding_dim).to(device)
decoder = CustomRNNDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(dataloader.dataset.vocab)).to(device)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

#BLEU score calculation
print("Starting training...")
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()

    total_loss = 0 

    for i, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        captions = batch['tokenized_caption'].to(device)

        features = encoder(images)
        outputs = decoder(features, captions)

        loss = criterion(outputs.view(-1, len(dataloader.dataset.vocab)), captions.view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {total_loss / len(dataloader):.4f}. Models saved.")

torch.save(encoder.state_dict(), f"custom_cnn_encoder.pth")
torch.save(decoder.state_dict(), f"custom_rnn_decoder.pth")

print("Training completed successfully!")






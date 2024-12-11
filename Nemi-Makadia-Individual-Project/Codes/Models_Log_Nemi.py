# Summary of Models Tried in the Project

# Model 1: ResNet50 + LSTM (Baseline Model)
from torchvision import models
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# Model 2: Vision Transformer (ViT) + Transformer Decoder
from transformers import ViTModel

class ViTEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ViTEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.fc = nn.Linear(self.vit.config.hidden_size, embed_size)

    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        features = self.fc(outputs.last_hidden_state[:, 0, :])
        return features

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=8, num_encoder_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        embeddings = embeddings.permute(1, 0, 2)
        features = features.unsqueeze(0)
        outputs = self.transformer(features, embeddings)
        outputs = self.fc(outputs)
        return outputs

# Model 3: Custom CNN + RNN
class CustomCNN(nn.Module):
    def __init__(self, embed_size):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 112 * 112, embed_size)

    def forward(self, images):
        x = self.conv1(images)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CustomRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.rnn(embeddings)
        outputs = self.fc(hiddens)
        return outputs

# Model 4: Vision Transformer Encoder + LSTM Decoder
class VisionTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(VisionTransformerEncoder, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
        self.fc = nn.Linear(768, embedding_dim)

    def forward(self, images):
        features = self.vit(images)
        embeddings = self.fc(features)
        return embeddings

class CaptionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.end_token_id = vocab("<end>")

    def forward(self, image_features, captions):
        embeddings = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
        lstm_output, _ = self.lstm(inputs)
        outputs = self.fc(lstm_output)
        return outputs

    def generate_caption(self, image_features, max_length=20):
        generated_ids = []
        states = None
        inputs = image_features.unsqueeze(1)
        for _ in range(max_length):
            lstm_output, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_output.squeeze(1))
            predicted_token = outputs.argmax(dim=1)
            generated_ids.append(predicted_token.item())
            if predicted_token.item() == self.end_token_id:
                break
            inputs = self.word_embeddings(predicted_token).unsqueeze(1)
        return generated_ids

# Model 5: BLIP-2
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIP2Model:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_captions(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
        return captions


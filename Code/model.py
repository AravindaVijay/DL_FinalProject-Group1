import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim, trainable_layers=2):
        super(ImageEncoder, self).__init__()
        base_model = models.resnet50(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False
        for param in list(base_model.parameters())[-trainable_layers:]:
            param.requires_grad = True
        feature_extractor = list(base_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.fc = nn.Linear(base_model.fc.in_features, embedding_dim)

    def forward(self, images):
        extracted_features = self.feature_extractor(images)
        extracted_features = extracted_features.view(extracted_features.size(0), -1)
        embeddings = self.fc(extracted_features)
        return embeddings

class CaptionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_state = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))

    def forward(self, image_features, captions):
        caption_embeddings = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat((image_features.unsqueeze(1), caption_embeddings), dim=1)
        lstm_output, self.hidden_state = self.lstm(inputs)
        outputs = self.fc(lstm_output)
        return outputs

    def generate_caption(self, inputs, states=None, max_length=20):
        generated_ids = []
        for _ in range(max_length):
            lstm_output, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_output.squeeze(1))
            predicted_token = outputs.argmax(dim=1)  
            generated_ids.append(predicted_token.item())
            if predicted_token == 1:
                break
            inputs = self.word_embeddings(predicted_token).unsqueeze(1)
        return generated_ids

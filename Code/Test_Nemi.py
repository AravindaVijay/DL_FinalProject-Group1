##########################################################################
######################PRETRAINED RESNET AND LSTM########################
##########################################################################

# from vocabulary import Vocabulary
# from model import ImageEncoder, CaptionDecoder
# from data_loader import get_data_loader
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import torch
# import numpy as np

# # Paths
# images_dir = '../COCO_Data/train2017'
# captions_path = '../COCO_Data/annotations/captions_train2017.json'
# encoder_path = "encoder_epoch_2.pth"
# decoder_path = "decoder_epoch_2.pth"

# # Load vocabulary
# vocab = Vocabulary(annotations_file=captions_path, vocab_exists=True)

# # Model parameters
# embedding_dim = 256
# hidden_dim = 512
# vocab_size = len(vocab)

# # Initialize models
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoder = ImageEncoder(embedding_dim=embedding_dim).to(device)
# decoder = CaptionDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, vocab=vocab).to(device)

# # Load saved models
# encoder.load_state_dict(torch.load(encoder_path))
# decoder.load_state_dict(torch.load(decoder_path))
# encoder.eval()
# decoder.eval()

# # Load data
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# dataloader = get_data_loader(images_dir, captions_path, vocab_exists=True, batch_size=1, transform=transform)

# # Generate captions for a few batches
# num_samples = 5  # Number of images to process
# sample_count = 0

# for batch in dataloader:
#     images = batch['images'].to(device)
#     image_ids = batch['image_ids']  # Get the image IDs
#     features = encoder(images)

#     # Generate captions
#     generated_ids = decoder.generate_caption(features)
#     generated_caption = [vocab.idx2word[idx] for idx in generated_ids]

#     # Get the corresponding image file name
#     image_info = dataloader.dataset.coco.loadImgs(image_ids[0])[0]  # Access image info from COCO
#     image_filename = image_info['file_name']

#     # Display image and generated caption
#     image = images[0].cpu().permute(1, 2, 0).numpy()
#     image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # De-normalize
#     image = np.clip(image, 0, 1)

#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(f"Generated Caption: {' '.join(generated_caption)}\nImage File: {image_filename}")
#     plt.show()

#     print(f"Image File: {image_filename}")
#     print(f"Generated Caption {sample_count+1}: {' '.join(generated_caption)}")
#     sample_count += 1

#     # Stop after processing num_samples images
#     if sample_count >= num_samples:
#         break


##########################################################################
###################### CNN AND RNN ########################
##########################################################################

from vocabulary import Vocabulary
from data_loader import get_data_loader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
from model import CustomCNNEncoder, CustomRNNDecoder


# Paths
images_dir = '../COCO_Data/train2017'
captions_path = '../COCO_Data/annotations/captions_train2017.json'
encoder_path = "encoder_epoch_2.pth"
decoder_path = "decoder_epoch_2.pth"

# Load vocabulary
vocab = Vocabulary(annotations_file=captions_path, vocab_exists=True)

# Model parameters
embedding_dim = 256
hidden_dim = 512
vocab_size = len(vocab)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = CustomCNNEncoder(embedding_dim=embedding_dim).to(device)
decoder = CustomRNNDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(vocab)).to(device)

# Load saved model weights
encoder.load_state_dict(torch.load("custom_encoder.pth"))
decoder.load_state_dict(torch.load("custom_decoder.pth"))
encoder.eval()
decoder.eval()

# Load data
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataloader = get_data_loader(images_dir, captions_path, vocab_exists=True, batch_size=1, transform=transform)

# Generate captions for a few batches
num_samples = 5  
sample_count = 0

for batch in dataloader:
    images = batch['images'].to(device)
    image_ids = batch['image_ids']  
    features = encoder(images)

    # Generate captions
    generated_ids = decoder.generate_caption(features)
    generated_caption = [vocab.idx2word[idx] for idx in generated_ids]

    # Get the corresponding image file name
    image_info = dataloader.dataset.coco.loadImgs(image_ids[0])[0]  
    image_filename = image_info['file_name']

    # Display image and generated caption
    image = images[0].cpu().permute(1, 2, 0).numpy()
    image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated Caption: {' '.join(generated_caption)}\nImage File: {image_filename}")
    plt.show()

    print(f"Image File: {image_filename}")
    print(f"Generated Caption {sample_count+1}: {' '.join(generated_caption)}")
    sample_count += 1

    # Stop after processing num_samples images
    if sample_count >= num_samples:
        break
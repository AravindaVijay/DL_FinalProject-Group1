from model_nihar import ImageEncoder, CaptionDecoder
from data_loader import get_data_loader
from torchvision import transforms
from tqdm.notebook import tqdm
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torch
from vocabulary import Vocabulary
import nltk


####################################
###### Defining the parameters #####
####################################


## Load the vocabulary
vocab = Vocabulary(annotations_file='../../coco_dataset/annotations/captions_train2017.json', vocab_exists=True)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64

# Define the transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the data loader
images_dir = '../../coco_dataset/train2017'
captions_path = '../../coco_dataset/annotations/captions_train2017.json'

dataloader = get_data_loader(
    images_dir=images_dir,
    captions_path=captions_path,
    vocab_exists=True,
    batch_size=batch_size,
    transform=transform
)


# Define the model
embedding_dim = 256
hidden_dim = 512
vocab_size = len(vocab)
encoder = ImageEncoder(embedding_dim).to(device)
decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size, vocab).to(device)

# Defining the loss function
criterion = (
    nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>']).cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
)
# criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])

# params = list(decoder.parameters()) + list(encoder.embed.parameters())

# Defining the optimize
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=5e-3)

# Set the learning rate scheduler
encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=5, gamma=0.1)
decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=5, gamma=0.1)


####################################
####### Training the model #########
####################################


# Train the model
num_epochs = 30

print("Starting training...")
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    epoch_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Load data to device
        images = batch['images'].to(device)
        captions = batch['tokenized_caption'].to(device).long()

        # Zero the parameter gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass
        image_features = encoder(images)
        outputs = decoder(image_features, captions)

        # Compute the loss
        targets = captions[:, 1:]  # Shift captions by one for target
        outputs = outputs[:, 1:, :]  # Exclude the prediction for the first token
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        # loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

        # Backward pass and optimization
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Step the scheduler
    encoder_scheduler.step()
    decoder_scheduler.step()

    # Print epoch loss
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save checkpoint after each epoch
    checkpoint_path = f"./checkpoints_nihar/efficientnet_epoch_{epoch + 1}.pth"
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
        'loss': avg_loss
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

print("Training completed.")




####################################
####### Evaluating the model #######
####################################


# Paths for saved models and validation data
checkpoint_path = './checkpoints_nihar/efficientnet_epoch_10.pth'
val_images_dir = '../../coco_dataset/val2017'
val_captions_path = '../../coco_dataset/annotations/captions_val2017.json'

# Load the model
encoder = ImageEncoder(embedding_dim).to(device)
decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size, vocab).to(device)

# Load the trained weightsencoder_state_dict
encoder.load_state_dict(torch.load(checkpoint_path, map_location=device)['encoder_state_dict'])
decoder.load_state_dict(torch.load(checkpoint_path, map_location=device)['decoder_state_dict'])

# Set models to evaluation mode
encoder.eval()
decoder.eval()


from collections import Counter
import math

def compute_bleu(reference_corpus, candidate_corpus, max_n=4, smoothing=True):
    """
    Compute BLEU score for a candidate corpus against a reference corpus.

    Args:
        reference_corpus (list of list of str): List of references (each a tokenized list of words).
        candidate_corpus (list of list of str): List of candidates (each a tokenized list of words).
        max_n (int): Maximum n-gram order to use.
        smoothing (bool): Apply smoothing to prevent zero scores for missing n-grams.

    Returns:
        float: The BLEU score.
    """
    def ngram_counts(tokens, n):
        """Generate n-gram counts for a list of tokens."""
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    
    clipped_counts = Counter()
    total_counts = Counter()
    reference_lengths = []
    candidate_lengths = []

    for references, candidate in zip(reference_corpus, candidate_corpus):
        candidate_lengths.append(len(candidate))
        reference_lengths.append(min(len(ref) for ref in references))

        for n in range(1, max_n + 1):
            candidate_ngrams = ngram_counts(candidate, n)
            reference_ngrams = Counter()
            for ref in references:
                reference_ngrams |= ngram_counts(ref, n)

            # Clip counts to reference max counts
            for ngram, count in candidate_ngrams.items():
                clipped_counts[ngram] += min(count, reference_ngrams[ngram])
            total_counts[n] += sum(candidate_ngrams.values())

    # Calculate precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            precisions.append(0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    # Smoothing: Replace 0 precision with a very small value to avoid log(0)
    if smoothing:
        precisions = [p if p > 0 else 1e-9 for p in precisions]

    # Geometric mean of precisions
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    # Brevity penalty
    ref_length = sum(reference_lengths)
    cand_length = sum(candidate_lengths)
    brevity_penalty = math.exp(1 - ref_length / cand_length) if cand_length < ref_length else 1

    return brevity_penalty * geometric_mean



# Define the evaluation function
def evaluate_model(dataloader, encoder, decoder, vocab, transform, device, max_length=20):
    references = []
    hypotheses = []

    for batch in tqdm(dataloader, desc='Evaluating', unit='batch'):
        images = batch['images'].to(device)
        all_captions = batch['all_captions']  # List of reference captions for each image

        # Generate caption for the image
        with torch.no_grad():
            image_features = encoder(images)
            generated_ids = decoder.generate_caption(image_features, max_length=max_length)
            generated_caption = [vocab.idx2word[idx] for idx in generated_ids if idx not in [vocab.start_word, vocab.end_word]]

        # Process reference captions
        reference_tokens = [
            [nltk.word_tokenize(caption.lower()) for caption in captions] 
            for captions in all_captions
        ]
        references.extend(reference_tokens)
        hypotheses.append(generated_caption)

    # Compute BLEU score
    avg_bleu_score = compute_bleu(references, hypotheses)
    return avg_bleu_score

# Validation dataloader
val_dataloader = get_data_loader(
    images_dir=val_images_dir,
    captions_path=val_captions_path,
    vocab_exists=True,
    batch_size=1,  # Evaluate one image at a time
    transform=transform
)

# Evaluate the model
avg_bleu_score = evaluate_model(val_dataloader, encoder, decoder, vocab, transform, device)
print(f"Average BLEU Score: {avg_bleu_score}")
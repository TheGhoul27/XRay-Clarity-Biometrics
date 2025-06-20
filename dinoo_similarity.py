import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_dinov2_patch_embeddings(image1_path, image2_path, model_name='dinov2_vits14'):
    """
    Extract patch embeddings from two images using DINOv2.
    
    Args:
        image1_path (str): Path to the first image
        image2_path (str): Path to the second image  
        model_name (str): DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
    
    Returns:
        tuple: (embeddings1, embeddings2) - patch embeddings for both images
               Each embedding tensor has shape [num_patches, embedding_dim]
    """
    
    # Load the DINOv2 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device)
    model.eval()
    
    # Define preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def process_image(image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract patch embeddings (not CLS token)
        with torch.no_grad():
            # Get all tokens (CLS + patches)
            features = model.forward_features(image_tensor)
            # Remove CLS token (first token) to get only patch embeddings
            patch_embeddings = features['x_norm_patchtokens']
            
        return patch_embeddings.squeeze(0).cpu()  # Remove batch dimension
    
    # Process both images
    embeddings1 = process_image(image1_path)
    embeddings2 = process_image(image2_path)
    
    return embeddings1, embeddings2

# Example usage
if __name__ == "__main__":
    # Extract patch embeddings
    img1 = "E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\AirPods Max\\frontside\\141849-V0_cls0_0\\best_reference.jpg"
    img2 = "E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\AirPods Max\\frontside\\141849-V0_cls0_0\\best_rotated.jpg"
    emb1, emb2 = get_dinov2_patch_embeddings(img1, img2)

    print(f"Image 1 patch embeddings shape: {emb1.shape}")
    print(f"Image 2 patch embeddings shape: {emb2.shape}")
    
    # Example: compute similarity between patch embeddings
    # Flatten and compute cosine similarity
    # emb1_flat = emb1.flatten()
    # emb2_flat = emb2.flatten()

    # print(f"Flattened patch embeddings shapes: {emb1_flat.shape}, {emb2_flat.shape}")
    
    # cosine_sim = torch.nn.functional.cosine_similarity(
    #     emb1_flat.unsqueeze(0), emb2_flat.unsqueeze(0)
    # )
    # print(f"Cosine similarity between patch embeddings: {cosine_sim.item():.4f}")

    dot_prod =  torch.matmul(emb1, emb2.T)
    print(f"Shape of dot product: {dot_prod.shape}")
    diagonal_values = torch.diagonal(dot_prod)
    patch_wise_similarity = diagonal_values / (torch.norm(emb1, dim=1) * torch.norm(emb2, dim=1))
    patch_wise_similarity = 1 - patch_wise_similarity
    plt.figure(figsize=(10, 5))
    plt.hist(patch_wise_similarity.numpy(), bins=50, color='blue')
    patch_wise_similarity = patch_wise_similarity.view(16, 16)
    print(f"Patch-wise similarity shape: {patch_wise_similarity.shape}")
    print(f"Patch-wise similarity values:\n{patch_wise_similarity}")
    patch_wise_overlay = cv2.resize(patch_wise_similarity.numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
    ax[0].imshow(cv2.resize(im1, (224, 224)))
    ax[1].imshow(cv2.resize(im2, (224, 224)))
    ax[1].imshow(patch_wise_overlay, cmap='hot', interpolation='nearest', alpha=0.3)
    # plt.colorbar(ax[1].imshow(patch_wise_overlay, cmap='hot', interpolation='nearest'))
    plt.show()

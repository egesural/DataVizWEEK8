import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# Page Configuration & Caching
# ---------------------------------------------------------
st.set_page_config(page_title="CIFAR-10 Image Analysis", layout="wide")
st.title("CIFAR-10 Image Browser & Saliency Analysis")

@st.cache_resource
def load_data():
    """Downloads and caches the CIFAR-10 dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes = dataset.classes
    return dataset, classes

@st.cache_resource
def load_model():
    """Loads a pre-trained ResNet18 model for Saliency extraction."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model, weights.transforms()

dataset, classes = load_data()
model, preprocess = load_model()

# Sidebar: App Navigation
st.sidebar.header("Navigation")
task = st.sidebar.radio(
    "Select Analysis Module:", 
    ["1. Dataset Montage & Balance", 
     "2. Structural Visualization", 
     "3. Saliency Map Viewer", 
     "4. Class Intensity Comparison"]
)

# ---------------------------------------------------------
# Module 1: Montage & Class Balance (Core + Extension 1)
# ---------------------------------------------------------
if task == "1. Dataset Montage & Balance":
    st.header("Dataset Montage & Class Balance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Image Montage")
        num_images = st.slider("Number of images in montage", 4, 64, 16, step=4)
        
        # Generate random indices
        indices = np.random.choice(len(dataset), num_images, replace=False)
        images = [dataset[i][0] for i in indices]
        
        # Create a grid using torchvision
        grid = torchvision.utils.make_grid(images, nrow=int(np.sqrt(num_images)))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        ax.axis('off')
        st.pyplot(fig)
        st.caption("This montage reveals the diverse orientations, lighting conditions, and low-resolution (32x32) nature of the CIFAR-10 dataset.")

    with col2:
        st.subheader("Class Balance")
        # CIFAR-10 is perfectly balanced (5000 per class in train), so we calculate it dynamically
        targets = np.array(dataset.targets)
        class_counts = {classes[i]: np.sum(targets == i) for i in range(len(classes))}
        st.bar_chart(class_counts)
        st.caption("Bar chart confirming the dataset is perfectly balanced across all 10 classes.")

# ---------------------------------------------------------
# Module 2: Structural Visualization (Core Task 1)
# ---------------------------------------------------------
elif task == "2. Structural Visualization":
    st.header("Structural Visualization: RGB Channels & Intensity")
    
    img_idx = st.number_input("Select an Image Index", min_value=0, max_value=len(dataset)-1, value=42)
    img_tensor, label = dataset[img_idx]
    img_np = np.transpose(img_tensor.numpy(), (1, 2, 0))
    
    st.subheader(f"Class: {classes[label].capitalize()}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(img_np, caption="Original Image", use_container_width=True)
    with col2:
        st.image(img_np[:, :, 0], caption="Red Channel", clamp=True, channels="GRAY", use_container_width=True)
    with col3:
        st.image(img_np[:, :, 1], caption="Green Channel", clamp=True, channels="GRAY", use_container_width=True)
    with col4:
        st.image(img_np[:, :, 2], caption="Blue Channel", clamp=True, channels="GRAY", use_container_width=True)

    st.subheader("RGB Intensity Histogram")
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist, bins = np.histogram(img_np[:, :, i], bins=256, range=(0.0, 1.0))
        ax.plot(bins[:-1], hist, color=color, alpha=0.7, label=f'{color.capitalize()} Channel')
    ax.set_title("Pixel Intensity Distribution per Channel")
    ax.set_xlabel("Pixel Intensity (0 to 1)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------
# Module 3: Saliency Map Viewer (Core + Extension 2)
# ---------------------------------------------------------
elif task == "3. Saliency Map Viewer":
    st.header("Feature Map & Saliency Overlay")
    
    upload = st.file_uploader("Upload your own image (or use a default CIFAR-10 image):", type=['jpg', 'png', 'jpeg'])
    
    if upload is not None:
        image = Image.open(upload).convert('RGB')
    else:
        img_idx = st.number_input("Or select a CIFAR-10 Image Index", min_value=0, max_value=len(dataset)-1, value=7)
        image = transforms.ToPILImage()(dataset[img_idx][0])
        st.write(f"**True Label:** {classes[dataset[img_idx][1]].capitalize()}")

    # Prepare image for ResNet18
    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor.requires_grad_()
    
    # Forward pass
    output = model(input_tensor)
    pred_class_idx = output.argmax().item()
    
    # Backward pass to get gradients
    model.zero_grad()
    score = output[0, pred_class_idx]
    score.backward()
    
    # Compute saliency map (maximum absolute gradient across color channels)
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    saliency = saliency[0].numpy()
    
    # Normalize saliency map for visualization
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    with col2:
        st.image(saliency, caption="Raw Saliency Map", clamp=True, channels="GRAY", use_container_width=True)
    with col3:
        # Overlay heatmap
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(saliency, cmap='hot', alpha=0.5)
        ax.axis('off')
        st.pyplot(fig)
        
    st.success(f"**Model Prediction Index (ImageNet):** {pred_class_idx}")
    st.markdown("""
    **Saliency Interpretation:** 
    The saliency map highlights the pixels that had the highest gradient magnitude with respect to the final predicted class score. The bright areas in the overlay heatmap represent the exact structural features (edges, textures, or shapes) that the neural network focused on to make its classification decision. If the heatmap is concentrated on the background, the model may be relying on spurious correlations rather than the main subject.
    """)

# ---------------------------------------------------------
# Module 4: Intensity Comparison (Extension 3)
# ---------------------------------------------------------
elif task == "4. Class Intensity Comparison":
    st.header("Compare Intensity Distributions Between Classes")
    
    col1, col2 = st.columns(2)
    with col1:
        class_a = st.selectbox("Select Class A", classes, index=0)
    with col2:
        class_b = st.selectbox("Select Class B", classes, index=1)
        
    if st.button("Generate Comparison"):
        with st.spinner(f"Computing average intensities for {class_a} and {class_b}..."):
            idx_a = classes.index(class_a)
            idx_b = classes.index(class_b)
            
            # Sample 100 images from each class to keep computation fast
            targets = np.array(dataset.targets)
            a_indices = np.where(targets == idx_a)[0][:100]
            b_indices = np.where(targets == idx_b)[0][:100]
            
            a_pixels = np.concatenate([dataset[i][0].numpy().flatten() for i in a_indices])
            b_pixels = np.concatenate([dataset[i][0].numpy().flatten() for i in b_indices])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(a_pixels, bins=100, alpha=0.5, label=class_a.capitalize(), density=True, color='blue')
            ax.hist(b_pixels, bins=100, alpha=0.5, label=class_b.capitalize(), density=True, color='orange')
            ax.set_title(f"Grayscale Pixel Intensity Distribution: {class_a.capitalize()} vs {class_b.capitalize()}")
            ax.set_xlabel("Pixel Intensity (0 to 1)")
            ax.set_ylabel("Density")
            ax.legend()
            
            st.pyplot(fig)
            st.caption(f"This histogram compares the aggregate pixel distributions of a 100-image sample from both classes. Differences in distribution shape can reveal if one class is inherently darker, lighter, or has higher contrast than the other.")
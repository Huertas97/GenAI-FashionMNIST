import gradio as gr
import json
import torch
import torchvision
from torch import nn
from diffusers import UNet2DModel, DDPMScheduler
import safetensors
from huggingface_hub import hf_hub_download

### GPU SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## LOAD THE UNET MODEL AND DDPM SCHEDULER FROM HUGGINGFACE HUB
class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=10, class_emb_size=10):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels
    # to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=28,           # output image resolution. Equal to input resolution
        in_channels=1 + class_emb_size, # Additional input channels for class cond
        out_channels=1,           # the number of output channels. Equal to input
        layers_per_block=3,       # three residual connections (ResNet) per block
        block_out_channels=(128, 256, 512), # N of output channels for each block. Inverse for upsampling
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        ),
        up_block_types=(
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "AttnUpBlock2D",
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
        dropout = 0.1,  # Dropout prob between Conv1 and Conv2 in a block. From Improved DDPM paper
    )

  # Forward method takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    bs, ch, w, h = x.shape # x is shape (bs, 1, 28, 28)

    # class conditioning embedding to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # class_cond final shape (bs, 4, 28, 28)

    # Model input is now x and class cond concatenated together along dimension 1
    # We need provide additional information (the class label)
    # to every spatial location (pixel) in the image. Not changing the original
    # pixels of the images, but adding new channels.
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    # with image output size
    return self.model(net_input, t).sample # (bs, 1, 28, 28)
  
# Define paths to download the model and scheduler
repo_name = "Huertas97/conditioned-unet-fashion-mnist-non-ema"

### UNET MODEL
# Download the safetensors model file
model_file_path = hf_hub_download(repo_id=repo_name, filename="fashion_class_cond_unet_model_best.safetensors")

# Load the Class Conditioned UNet model state dictionary
state_dict = safetensors.torch.load_file(model_file_path)
model_classcond_native =  ClassConditionedUnet()
model_classcond_native.load_state_dict(state_dict)
model_classcond_native.to(device)

### DDPM SCHEDULER
# Download and load the scheduler configuration file
scheduler_file_path = hf_hub_download(repo_id=repo_name, filename="scheduler_config.json")

with open(scheduler_file_path, 'r') as f:
    scheduler_config = json.load(f)

noise_scheduler = DDPMScheduler.from_config(scheduler_config)




# Define the classes
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def generate_images(selected_class, num_images, progress=gr.Progress()):
    """
    Generate images using the trained model.
    
    Parameters:
    - selected_class: The class label as a string.
    - num_images: Number of images to generate.
    
    Returns:
    - A list of generated images.
    """
    # Convert class label to corresponding index
    class_idx = class_labels.index(selected_class)
    
    # Prepare random x to start from
    x = torch.randn(num_images, 1, 28, 28).to(device)
    y = torch.tensor([class_idx] * num_images).to(device)
    
    for t in progress.tqdm(noise_scheduler.timesteps, desc="Generating image", total=noise_scheduler.config.num_train_timesteps): # 
        with torch.no_grad():
            residual = model_classcond_native(x, t, y)
        x = noise_scheduler.step(residual, t, x).prev_sample

    # Post-process the generated images
    # Clamp the values to [0, 1] and convert to [0, 255] uint8
    # Also move the tensor to CPU and convert to numpy for plotting
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8).cpu()
    
    # Convert to list of images
    images = [img.squeeze(0).numpy() for img in x]
    return images

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Dropdown(class_labels, label="Select Class", value="T-shirt/top"),
        gr.Slider(minimum=1, maximum=8, step=1, value=1, label="Number of Images")
    ],
    outputs=gr.Gallery(type="numpy", label="Generated Images"),
    live=False,
    description="Generate images using a class-conditioned UNet model."
)

demo.launch(share=True)
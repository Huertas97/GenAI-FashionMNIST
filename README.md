# GenAI-FashionMNIST
Github repository with the implementation of a diffusion UNet model able to generate synthetic images of clothes from the FashionMNIST dataset

The code can be found in `notebooks` folder. 
The Diffusion model generates synthetic images of clothes from the FashionMNIST dataset, conditioned by the class of the image. 
The tracked results of the training experiments can be found on the following W&B [project](https://wandb.ai/huertas_97/GenAI-FashionMNIST). 


## Model and Demo

The model can be found in HuggingFace's model hub [here](https://huggingface.co/Huertas97/conditioned-unet-fashion-mnist-non-ema).

In the `notebooks` folder you can find the `gradio_demo.py` code for running a demo of the model. Here a screenshot of the demo:

![2024-07-16 11_18_06-Gradio](https://github.com/user-attachments/assets/142cc658-0aef-4136-a8d4-1c0310e60a31)


## Results
Here are some generated images:

![image](https://github.com/user-attachments/assets/e3d390c6-2a9c-486d-9fb6-c3a7cd4fc841)





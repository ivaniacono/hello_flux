import torch
from diffusers import FluxPipeline
from huggingface_hub import snapshot_download
import os 

torch.cuda.empty_cache()

model="FLUX.1-schnell"
repo = "black-forest-labs/" + model

dl_dir = os.getcwd() + "/dl/" + model

snapshot_download(repo_id=repo, local_dir=dl_dir)
pipe = FluxPipeline.from_pretrained(dl_dir, torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload() # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.enable_sequential_cpu_offload() # offload modules to CPU on a submodule level rather then on a module level

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-out.png")

torch.cuda.empty_cache()
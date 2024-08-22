# Hello Flux

Python test environment and example script to run FLUX.1 locally on a GPU with limited VRAM.

Tested both schnell and dev on linux with a Nvidia 4060 8GB GPU.

(Note: When changing schnell for dev, a Hugging Face token will have to be passed at runtime)

## Run

```
python3 -m venv venv

. venv/bin/activate

pip install -r requirements.txt

python hello_flux.py
```
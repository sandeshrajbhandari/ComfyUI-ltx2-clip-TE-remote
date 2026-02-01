# ComfyUI LTXV Remote Text Encoder

This custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) allows you to use a **remote LTXV (Lightweight Text-to-Video) text encoder** hosted on Hugging Face spaces. This is particularly useful for users with limited VRAM who want to offload the heavy text encoding process (typically using Gemma or T5) to a remote server.

## Features

- **Remote Encoding**: Connects to a Hugging Face Space (default: `linoyts-gemma-text-encoder`) to generate embeddings.
- **Local Caching**: Caches generated embeddings locally to avoid redundant network requests and speed up workflow iterations.
- **Automatic Dependencies**: Automatically attempts to install required Python packages (`requests`, `huggingface_hub`) if missing.
- **Load Pre-computed Embeddings**: Includes a node to load existing `.pt` embedding files directly.

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/sandeshrajbhandari/ComfyUI-ltx2-clip-TE-remote.git
    ```
3.  Install dependencies (optional, as the node attempts to auto-install):
    ```bash
    cd ComfyUI-ltx2-clip-TE-remote
    pip install -r requirements.txt
    ```

## Usage

### Remote LTXV Text Encoder (HF)
This is the main node. It takes a prompt and returns positive and negative conditioning.

- **Inputs**:
    - `prompt`: The positive text prompt.
    - `negative_prompt`: The negative text prompt.
    - `enhance_prompt`: Boolean to enable prompt enhancement (if supported by the remote).
    - `seed`: Seed for reproducibility.
    - `hf_endpoint_url`: (Optional) Custom HF Gradio endpoint. Defaults to `https://linoyts-gemma-text-encoder.hf.space/gradio_api/call/encode_prompt`.
    - `hf_api_token`: (Optional) Your Hugging Face Read Token. 
    - `device`: Device to load the final embedding onto (`cpu` or `cuda`).

- **Outputs**:
    - `positive`: CONDITIONING for the sampler.
    - `negative`: CONDITIONING for the sampler.

### Load LTXV Embedding (.pt)
Use this node to load a specific embedding file you may have saved or downloaded.

- **Input**:
    - `file_path`: Path to the `.pt` file.
- **Outputs**:
    - `positive` / `negative` conditioning.

## Configuration

You can set environment variables to avoid entering them in the node every time:

- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`: Your Hugging Face API token.
- `HF_LTXV_REMOTE_ENCODER_URL`: Default URL for the remote encoder.

## Caching
Embeddings are cached in `ComfyUI/cache/ltxv_remote_cache`. You can manage this folder to clear old cache files if needed.

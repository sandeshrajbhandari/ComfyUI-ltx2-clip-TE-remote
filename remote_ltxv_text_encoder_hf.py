# --- Auto-install light dependencies if missing (requests, huggingface_hub) ---
def _ensure_deps():
    import importlib
    missing = []
    for pkg in ("requests", "huggingface_hub"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        import subprocess, sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except Exception as e:
            print(f"[RemoteLTXVTextEncoderHF] Failed to auto-install {missing}: {e}")

_ensure_deps()
# --- End auto-install block ---

import io
import json
import os
import time
import hashlib
import torch
import requests
import folder_paths
from huggingface_hub import get_token
import torch.nn.functional as F

HF_DEFAULT_URL = os.environ.get(
    "HF_LTXV_REMOTE_ENCODER_URL",
    "https://linoyts-gemma-text-encoder.hf.space/gradio_api/call/encode_prompt",
)

class CacheManager:
    def __init__(self, cache_dir="ltxv_remote_cache", max_size=50):
        self.cache_dir = os.path.join(folder_paths.base_path, "cache", cache_dir)
        self.manifest_path = os.path.join(self.cache_dir, "cache_manifest.json")
        self.max_size = max_size
        self.cache_data = {} # Key -> {timestamp, pos_file, neg_file}
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.load_manifest()

    def load_manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r") as f:
                    self.cache_data = json.load(f)
            except Exception as e:
                print(f"[CacheManager] Failed to load manifest: {e}")
                self.cache_data = {}

    def save_manifest(self):
        try:
            with open(self.manifest_path, "w") as f:
                json.dump(self.cache_data, f, indent=2)
        except Exception as e:
            print(f"[CacheManager] Failed to save manifest: {e}")

    def get(self, key):
        if key in self.cache_data:
            entry = self.cache_data[key]
            pos_path = os.path.join(self.cache_dir, entry["pos_file"])
            neg_path = os.path.join(self.cache_dir, entry["neg_file"])
            
            if os.path.exists(pos_path) and os.path.exists(neg_path):
                try:
                    # Update timestamp for LRU
                    entry["timestamp"] = time.time()
                    self.save_manifest()
                    
                    pos_emb = torch.load(pos_path, map_location="cpu")
                    neg_emb = torch.load(neg_path, map_location="cpu")
                    return {"positive": pos_emb, "negative": neg_emb}
                except Exception as e:
                    print(f"[CacheManager] Error loading cached files: {e}")
                    if key in self.cache_data: del self.cache_data[key]
                    self.save_manifest()
                    return None
            else:
                if key in self.cache_data: del self.cache_data[key]
                self.save_manifest()
                return None
        return None

    def set(self, key, pos_emb, neg_emb):
        try:
            if len(self.cache_data) >= self.max_size:
                oldest_key = min(self.cache_data, key=lambda k: self.cache_data[k]["timestamp"])
                self.delete_entry(oldest_key)

            timestamp = time.time()
            pos_filename = f"{key}_pos.pt"
            neg_filename = f"{key}_neg.pt"
            
            pos_path = os.path.join(self.cache_dir, pos_filename)
            neg_path = os.path.join(self.cache_dir, neg_filename)
            
            torch.save(pos_emb, pos_path)
            torch.save(neg_emb, neg_path)
            
            self.cache_data[key] = {
                "timestamp": timestamp,
                "pos_file": pos_filename,
                "neg_file": neg_filename
            }
            self.save_manifest()
        except Exception as e:
            print(f"[CacheManager] Error saving to cache: {e}")

    def delete_entry(self, key):
        if key in self.cache_data:
            entry = self.cache_data[key]
            pos_path = os.path.join(self.cache_dir, entry["pos_file"])
            neg_path = os.path.join(self.cache_dir, entry["neg_file"])
            if os.path.exists(pos_path): os.remove(pos_path)
            if os.path.exists(neg_path): os.remove(neg_path)
            del self.cache_data[key]


def call_ltxv_remote_encoder(
    prompt,
    negative_prompt="",
    enhance_prompt=True,
    seed=42,
    endpoint_url=None,
    api_token=None,
    device="cpu",
    timeout=120,
):
    """
    Calls the remote LTXV Gemma text encoder on Hugging Face.
    """
    payload = {
        "data": [
            prompt,
            enhance_prompt,
            None,  # input_image
            seed,
            negative_prompt,
        ]
    }

    url = endpoint_url or HF_DEFAULT_URL
    token = (
        api_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or get_token()
    )

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("[RemoteLTXVTextEncoderHF] Using Hugging Face API Token: Yes (Masked)")
    else:
        print("[RemoteLTXVTextEncoderHF] Using Hugging Face API Token: No")

    print(f"[RemoteLTXVTextEncoderHF] Making initial POST request to: {url}")
    post_resp = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )
    post_resp.raise_for_status()

    post_data = post_resp.json()
    event_id = post_data.get("event_id") or post_data.get("data")
    if event_id is None:
        raise ValueError(f"[RemoteLTXVTextEncoderHF] Could not extract EVENT_ID: {post_resp.text}")

    get_url = f"{url}/{event_id}"
    print(f"[RemoteLTXVTextEncoderHF] Making subsequent GET request to: {get_url}")

    response_json = None
    try:
        # The API returns Server-Sent Events (SSE), so we must stream the response.
        with requests.get(get_url, headers=headers, stream=True, timeout=timeout) as get_resp:
            get_resp.raise_for_status()
            
            for line in get_resp.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # print(f"DEBUG: Received line: {decoded_line}") 
                    # Look for data lines. Example: "data: [...]"
                    if decoded_line.startswith('data:'):
                        json_str = decoded_line[5:].strip()
                        if not json_str or json_str == "null":
                            continue
                        
                        try:
                            data = json.loads(json_str)
                            print(f"[RemoteLTXVTextEncoderHF] Parsed data: {str(data)[:100]}...") # Truncate for log safety
                            # The success data is a list: [{file_info}, "prompt", "status"]
                            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                                if "url" in data[0] or "path" in data[0]:
                                    response_json = {"data": data}
                                    break
                        except json.JSONDecodeError:
                            print(f"[RemoteLTXVTextEncoderHF] Failed to decode JSON from line: {decoded_line[:50]}...")
                            continue

    except Exception as e:
        print(f"[RemoteLTXVTextEncoderHF] Error during GET request: {e}")

    if response_json is None:
        raise ValueError("[RemoteLTXVTextEncoderHF] Did not receive valid data from the server (checked for SSE 'data:' lines).")

    download_url = response_json["data"][0].get('url') if isinstance(response_json["data"][0], dict) else None
    if download_url is None: return None, None

    print(f"[RemoteLTXVTextEncoderHF] Downloading embeddings from: {download_url}")
    embedding_resp = requests.get(download_url, timeout=timeout)
    embedding_resp.raise_for_status()

    embedding_data = torch.load(io.BytesIO(embedding_resp.content), map_location="cpu")
    video_context = embedding_data.get("video_context")
    audio_context = embedding_data.get("audio_context")

    if video_context is not None and audio_context is not None:
        # Concatenate Video (3840) + Audio (3840) = 7680
        full_context = torch.cat([video_context, audio_context], dim=-1)
        print(f"[RemoteLTXVTextEncoderHF] Concatenated Video + Audio. Shape: {full_context.shape}")
    elif video_context is not None:
        full_context = video_context
        print(f"[RemoteLTXVTextEncoderHF] Only Video context found. Shape: {full_context.shape}")
    else:
        return None, None

    if device != "cpu": full_context = full_context.to(device)
    return full_context, None

def _make_cond_from_embedding(e: torch.Tensor | None, like: torch.Tensor | None = None):
    if e is None:
        if like is not None:
            z = torch.zeros_like(like)
        else:
            z = torch.zeros((1, 1, 7680), dtype=torch.float32)
    else:
        z = e

    if z.dtype != torch.float32: z = z.to(torch.float32)
    if z.ndim == 2: z = z.unsqueeze(1)

    return [[z, {"pooled_output": None}]]


class RemoteLTXVTextEncoderHF:
    def __init__(self):
        self.cache_manager = CacheManager()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "hf_endpoint_url": ("STRING", {"default": HF_DEFAULT_URL}),
                "hf_api_token": ("STRING", {"default": ""}),
                "device": (["cpu", "cuda:0", "cuda:1"], {"default": "cpu"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "conditioning/ltxv_remote"

    def encode(self, prompt, negative_prompt="", enhance_prompt=True, seed=42, hf_endpoint_url=None, hf_api_token="", device="cpu"):
        cache_key = hashlib.sha256(f"{prompt}_{negative_prompt}_{enhance_prompt}_{seed}".encode()).hexdigest()
        cache_data = self.cache_manager.get(cache_key)
        
        if cache_data:
            print(f"[RemoteLTXVTextEncoderHF] Cache hit: {cache_key}")
            pos_emb, neg_emb = cache_data["positive"], cache_data["negative"]
            if device != "cpu":
                pos_emb, neg_emb = pos_emb.to(device), neg_emb.to(device)
        else:
            if not prompt.strip() and not negative_prompt.strip():
                empty = _make_cond_from_embedding(None)
                return (empty, empty)

            pos_emb, _ = call_ltxv_remote_encoder(prompt, negative_prompt, enhance_prompt, seed, hf_endpoint_url, hf_api_token or None, device)
            if pos_emb is None: return ([], [])
            neg_emb = torch.zeros_like(pos_emb)
            self.cache_manager.set(cache_key, pos_emb.cpu(), neg_emb.cpu())

        return (_make_cond_from_embedding(pos_emb), _make_cond_from_embedding(neg_emb, like=pos_emb))


class LoadLTXVEmbedding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "ltxv_text_encoder_api/embedding.pt"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "load_embedding"
    CATEGORY = "conditioning/ltxv_remote"

    def load_embedding(self, file_path):
        print(f"[LoadLTXVEmbedding] Loading embedding from: {file_path}")
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        
        data = torch.load(file_path, map_location="cpu")
        if not isinstance(data, dict): raise ValueError("File must be a dictionary.")

        vid = data.get("video_context")
        aud = data.get("audio_context")
        neg = data.get("video_context_negative")

        if vid is not None and aud is not None:
            pos_emb = torch.cat([vid, aud], dim=-1)
            print(f"[LoadLTXVEmbedding] Concatenated Video+Audio context. Shape: {pos_emb.shape}")
        else:
            pos_emb = vid
            
        if pos_emb is None: raise ValueError("No valid embedding found in file.")
        
        neg_emb = neg if neg is not None else torch.zeros_like(pos_emb)
        
        return (_make_cond_from_embedding(pos_emb), _make_cond_from_embedding(neg_emb, like=pos_emb))


NODE_CLASS_MAPPINGS = {
    "RemoteLTXVTextEncoderHF": RemoteLTXVTextEncoderHF,
    "LoadLTXVEmbedding": LoadLTXVEmbedding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteLTXVTextEncoderHF": "Remote LTXV Text Encoder (HF)",
    "LoadLTXVEmbedding": "Load LTXV Embedding (.pt)",
}
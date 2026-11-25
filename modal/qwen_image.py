# Qwen-Image (Apache 2.0, strong in-image text) on a Modal H100. job.json: {"items": [{"name", "prompt", "w", "h", "seed"}], "neg": "..."}
import json, os, io, modal
app = modal.App("pp-qwen-image")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.6.0", "transformers>=4.51", "accelerate", "sentencepiece", "pillow", "huggingface_hub[hf_transfer]")
       .pip_install("git+https://github.com/huggingface/diffusers").env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"}))
vol = modal.Volume.from_name("pp-hf-cache", create_if_missing=True)

@app.cls(image=img, gpu="H100", volumes={"/cache": vol}, timeout=3000, scaledown_window=60)
class Q:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import DiffusionPipeline
        self.pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16, cache_dir="/cache").to("cuda")
        vol.commit()
    @modal.method()
    def draw(self, items, neg):
        import torch
        out = []
        for it in items:
            im = self.pipe(prompt=it["prompt"], negative_prompt=neg, width=it["w"], height=it["h"], num_inference_steps=it.get("steps", 40),
                           true_cfg_scale=it.get("cfg", 4.0), generator=torch.Generator("cuda").manual_seed(it["seed"])).images[0]
            b = io.BytesIO(); im.save(b, "JPEG", quality=95); out.append((it["name"], b.getvalue())); print(it["name"], "ok", flush=True)
        return out

@app.local_entrypoint()
def main(job: str = "job.json", out: str = "out"):
    J = json.load(open(job)); os.makedirs(out, exist_ok=True)
    for name, data in Q().draw.remote(J["items"], J.get("neg", " ")): open(os.path.join(out, name + ".jpg"), "wb").write(data)
    print("done", len(os.listdir(out)))

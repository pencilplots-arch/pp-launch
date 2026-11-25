# Z-Image-Turbo (Apache 2.0, 6B, good in-image text, 8-9 steps) on Modal CPUs (32 cores each): Modal's no-card plan refuses every GPU (T4 included, 25 Nov 2026).
# job.json: {"items": [{"name", "prompt", "w", "h", "seed"}], "parallel": 3}
import json, os, io, modal
app = modal.App("pp-zimage")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch>=2.8", "transformers>=4.51", "accelerate", "sentencepiece", "pillow", "huggingface_hub[hf_transfer]", "diffusers>=0.36")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"}))
vol = modal.Volume.from_name("pp-hf-cache", create_if_missing=True)

@app.cls(image=img, cpu=32.0, memory=65536, volumes={"/cache": vol}, timeout=3600, scaledown_window=60)
class Z:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import ZImagePipeline
        self.pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16, cache_dir="/cache")
        torch.set_num_threads(32)
        vol.commit()
    @modal.method()
    def draw(self, items):
        import torch
        out = []
        for it in items:
            im = self.pipe(prompt=it["prompt"], width=it["w"], height=it["h"], num_inference_steps=it.get("steps", 9), guidance_scale=0.0,
                           generator=torch.Generator("cpu").manual_seed(it["seed"])).images[0]
            b = io.BytesIO(); im.save(b, "JPEG", quality=95); out.append((it["name"], b.getvalue())); print(it["name"], "ok", flush=True)
        return out

@app.local_entrypoint()
def main(job: str = "job.json", out: str = "out"):
    J = json.load(open(job)); os.makedirs(out, exist_ok=True)
    n = J.get("parallel", 3); items = J["items"]; chunks = [items[i::n] for i in range(n)]
    for res in Z().draw.map([c for c in chunks if c]):
        for name, data in res: open(os.path.join(out, name + ".jpg"), "wb").write(data)
    print("done", len(os.listdir(out)))

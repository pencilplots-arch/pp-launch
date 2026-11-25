# FLUX.2 klein 4B on a Modal GPU. job.json: {"gpu": "L4", "items": [{"name", "prompt", "w", "h", "seed", "refs": ["refs/a.jpg", ...]}]}
# The model weights are cached in a Modal Volume, so only the first run downloads them.
import json, os, io, modal
app = modal.App("pp-flux-klein")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch>=2.8", "transformers", "accelerate", "sentencepiece", "pillow", "huggingface_hub")
       .pip_install("diffusers>=0.36"))
vol = modal.Volume.from_name("pp-hf-cache", create_if_missing=True)

@app.cls(image=img, gpu="L4", volumes={"/cache": vol}, timeout=1800, scaledown_window=60)
class Klein:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import Flux2KleinPipeline
        os.environ["HF_HOME"] = "/cache"
        self.pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16, cache_dir="/cache").to("cuda")
        vol.commit()
    @modal.method()
    def draw(self, items, refs):
        import torch
        from PIL import Image
        R = {k: Image.open(io.BytesIO(v)).convert("RGB") for k, v in refs.items()}
        out = []
        for it in items:
            kw = dict(prompt=it["prompt"], num_inference_steps=it.get("steps", 4), guidance_scale=1.0, width=it["w"], height=it["h"],
                      generator=torch.Generator("cuda").manual_seed(it["seed"]))
            if it.get("refs"): kw["image"] = [R[r] for r in it["refs"]]
            b = io.BytesIO(); self.pipe(**kw).images[0].save(b, "JPEG", quality=94); out.append((it["name"], b.getvalue())); print(it["name"], "ok", flush=True)
        return out

@app.local_entrypoint()
def main(job: str = "job.json", out: str = "out"):
    J = json.load(open(job)); os.makedirs(out, exist_ok=True)
    names = sorted({r for it in J["items"] for r in it.get("refs", [])})
    refs = {r: open(r, "rb").read() for r in names}
    n = J.get("parallel", 1); items = J["items"]; chunks = [items[i::n] for i in range(n)]
    for res in Klein().draw.starmap([(c, refs) for c in chunks if c]):
        for name, data in res: open(os.path.join(out, name + ".jpg"), "wb").write(data)
    print("done", len(os.listdir(out)))

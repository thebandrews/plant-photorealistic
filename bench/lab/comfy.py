"""Run ComfyUI workflows on the lab's RunPod endpoint (photoreal-diffusion).

Standard library only. The RunPod key comes from Key Vault (managed identity on the VM via the render
client, or `az` on a workstation) and is never printed or written. Every job's GPU seconds go to the lab
experiment's ledger (PLANT_GPU_BUDGET_FILE, the same file render-remote records into), and a job is refused
when the experiment budget (PLANT_GPU_BUDGET_SECONDS) or the lab budget (PHOTOREAL_LAB_GPU_SECONDS, summed over
every experiment's ledger) is spent.

    from lab import comfy
    imgs, info = comfy.run(workflow, {'in.png': 'path/to/in.png'})   # -> [PIL.Image], job info

Models on the endpoint's volume (/runpod-volume/models; the worker reads unet/ and clip/, not diffusion_models/ and text_encoders/): see MODELS.
"""
import base64
import datetime as dt
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = os.environ.get('PHOTOREAL_ENDPOINT', 'm9ujz947jujc1d')
API = 'https://api.runpod.ai/v2'
VAULT = 'kv-render-t3glhdrvtdeag'
RENDER_CLIENT = r'C:\Workstation\devbox-gpu\render\client'
ADHOC_LEDGER = os.environ.get('PHOTOREAL_ADHOC_LEDGER', r'C:\Workstation\lab\adhoc-gpu-ledger.json')
MAX_PAYLOAD = 9_500_000      # RunPod /run accepts 10 MB

MODELS = {
    'unet': ['qwen_image_edit_2511_fp8mixed.safetensors', 'qwen_image_edit_2509_fp8_e4m3fn.safetensors',
                         'qwen_image_2512_fp8_e4m3fn.safetensors'],
    'clip': ['qwen_2.5_vl_7b_fp8_scaled.safetensors'],
    'vae': ['qwen_image_vae.safetensors'],
    'controlnet': ['Qwen-Image-InstantX-ControlNet-Union.safetensors',
                   'Qwen-Image-InstantX-ControlNet-Inpainting.safetensors'],
    'loras': ['Qwen-Image-Edit-2509-Anything2RealAlpha.safetensors', 'anything2real_2601.safetensors',
              'anything2real_2601_A_final_patched.safetensors', 'Qwen-Image-Edit-2509-Relight.safetensors',
              'Qwen-Image-Edit-2509-Light-Migration.safetensors',
              'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors',
              'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
              'Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors',
              'Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors'],
}


class BudgetError(RuntimeError):
    pass


def _key():
    if os.path.isdir(RENDER_CLIENT):
        sys.path.insert(0, RENDER_CLIENT)
        import render_remote as rr
        return rr.Azure(rr.load_config()).secret('runpod-api-key')
    az = 'az.cmd' if os.name == 'nt' else 'az'
    return subprocess.run([az, 'keyvault', 'secret', 'show', '--vault-name', VAULT, '-n', 'runpod-api-key',
                           '--query', 'value', '-o', 'tsv'], capture_output=True, text=True, check=True).stdout.strip()


def _call(method, path, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f'{API}/{ENDPOINT}{path}', data=data, method=method,
                                 headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'RunPod {method} {path}: HTTP {e.code} {e.read()[:300]!r}') from None


def _ledger(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else {'entries': []}


def _used(path):
    return sum(e.get('gpu_seconds', 0) for e in _ledger(path)['entries'])


def _record(path, entry):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _ledger(p)
    data['entries'].append(entry)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, p)


def lab_used():
    """GPU seconds over every experiment's ledger (render-remote and diffusion jobs) plus ad-hoc jobs."""
    root = os.environ.get('PHOTOREAL_EXPERIMENTS')
    files = list(Path(root).parent.glob('experiments*/*/gpu-ledger.json')) if root and Path(root).is_dir() else []
    return sum(_used(f) for f in files + [Path(ADHOC_LEDGER)])


def _ledger_path():
    return os.environ.get('PLANT_GPU_BUDGET_FILE') or ADHOC_LEDGER


def check_budget(reserve=120):
    exp_limit = float(os.environ.get('PLANT_GPU_BUDGET_SECONDS', 0) or 0)
    if exp_limit and os.environ.get('PLANT_GPU_BUDGET_FILE'):
        used = _used(os.environ['PLANT_GPU_BUDGET_FILE'])
        if used + reserve > exp_limit:
            raise BudgetError(f'Experiment GPU budget spent: {used:.0f}s of {exp_limit:.0f}s')
    lab_limit = float(os.environ.get('PHOTOREAL_LAB_GPU_SECONDS', 0) or 0)
    if lab_limit and lab_used() + reserve > lab_limit:
        raise BudgetError(f'Lab GPU budget spent: {lab_used():.0f}s of {lab_limit:.0f}s')


def encode(path_or_image, fmt='PNG'):
    from PIL import Image
    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


def run(workflow, images=None, timeout=1800, label='comfy', poll=5):
    """Submit an API-format workflow with input images ({name: path or PIL image}); return (images, info)."""
    from PIL import Image
    check_budget()
    payload = {'workflow': workflow,
               'images': [{'name': n, 'image': encode(v)} for n, v in (images or {}).items()]}
    size = len(json.dumps(payload))
    if size > MAX_PAYLOAD:
        raise ValueError(f'payload {size / 1e6:.1f} MB is over the 10 MB RunPod limit; send smaller tiles')
    key = _key()
    t0 = time.time()
    job = _call('POST', '/run', key, {'input': payload})
    jid, st = job['id'], job
    try:
        while st.get('status') not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
            if time.time() - t0 > timeout:
                _call('POST', f'/cancel/{jid}', key)
                raise TimeoutError(f'job {jid} exceeded {timeout}s')
            time.sleep(poll)
            st = _call('GET', f'/status/{jid}', key)
    finally:
        gpu = round((st.get('executionTime') or 0) / 1000, 1)
        entry = {'job_id': jid, 'label': label, 'gpu_seconds': gpu, 'status': st.get('status'),
                 'queue_seconds': round((st.get('delayTime') or 0) / 1000, 1),
                 'at': dt.datetime.now().isoformat(timespec='seconds'),
                 'experiment': os.environ.get('PHOTOREAL_EXPERIMENT_ID')}
        _record(_ledger_path(), entry)
        key = None
    if st.get('status') != 'COMPLETED':
        raise RuntimeError(f"job {jid} {st.get('status')}: {json.dumps(st.get('error') or st.get('output'))[:1500]}")
    out = st.get('output') or {}
    imgs = []
    for item in out.get('images') or []:
        if item.get('type', 'base64') == 'base64':
            imgs.append(Image.open(io.BytesIO(base64.b64decode(item['data']))).convert('RGB'))
    if not imgs:
        raise RuntimeError(f'job {jid} returned no images: {json.dumps(out)[:500]}')
    return imgs, entry


# ---------------------------------------------------------------------------
# Workflow builders (ComfyUI API format). Experiments may build their own.
# ---------------------------------------------------------------------------
def qwen_edit(image_name, prompt, *, model='qwen_image_edit_2511_fp8mixed.safetensors', loras=(),
              steps=8, cfg=1.0, seed=1, megapixels=1.0, negative='', shift=3.1, denoise=1.0,
              reference_names=()):
    """Qwen-Image-Edit: one input image (+ up to 2 references), prompt -> edited image at ~megapixels."""
    wf = {
        'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': model, 'weight_dtype': 'default'}},
        'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen_2.5_vl_7b_fp8_scaled.safetensors',
                                                         'type': 'qwen_image', 'device': 'default'}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
        'load': {'class_type': 'LoadImage', 'inputs': {'image': image_name}},
        'scale': {'class_type': 'ImageScaleToTotalPixels',
                  'inputs': {'image': ['load', 0], 'upscale_method': 'lanczos', 'megapixels': megapixels, 'resolution_steps': 16}},
    }
    model_ref = ['unet', 0]
    for i, (name, strength) in enumerate(loras):
        wf[f'lora{i}'] = {'class_type': 'LoraLoaderModelOnly',
                          'inputs': {'model': model_ref, 'lora_name': name, 'strength_model': strength}}
        model_ref = [f'lora{i}', 0]
    wf['ms'] = {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': model_ref, 'shift': shift}}
    enc = {'clip': ['clip', 0], 'vae': ['vae', 0], 'image1': ['scale', 0]}
    for i, ref in enumerate(reference_names[:2], start=2):
        wf[f'ref{i}'] = {'class_type': 'LoadImage', 'inputs': {'image': ref}}
        enc[f'image{i}'] = [f'ref{i}', 0]
    wf['pos'] = {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': dict(enc, prompt=prompt)}
    wf['neg'] = {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': dict(enc, prompt=negative)}
    wf['latent'] = {'class_type': 'VAEEncode', 'inputs': {'pixels': ['scale', 0], 'vae': ['vae', 0]}}
    wf['sample'] = {'class_type': 'KSampler', 'inputs': {
        'model': ['ms', 0], 'positive': ['pos', 0], 'negative': ['neg', 0], 'latent_image': ['latent', 0],
        'seed': seed, 'steps': steps, 'cfg': cfg, 'sampler_name': 'euler', 'scheduler': 'simple',
        'denoise': denoise}}
    wf['decode'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['sample', 0], 'vae': ['vae', 0]}}
    wf['save'] = {'class_type': 'SaveImage', 'inputs': {'images': ['decode', 0], 'filename_prefix': 'out'}}
    return wf


if __name__ == '__main__':      # smoke test: python comfy.py in.png out.png
    src, dst = sys.argv[1], sys.argv[2]
    wf = qwen_edit('in.png', 'change the picture 1 to realistic photograph',
                   loras=[('Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors', 1.0),
                          ('anything2real_2601.safetensors', 0.8)])
    imgs, info = run(wf, {'in.png': src}, label='smoke')
    imgs[0].save(dst)
    print(json.dumps(info))

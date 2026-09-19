import shutil, subprocess, sys, time, urllib.request
print("nvidia-smi:", "present" if shutil.which("nvidia-smi") else "MISSING")
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
try:
    import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e: print("torch import failed", e)
try:
    urllib.request.urlopen("https://pypi.org/simple/rtmlib/", timeout=15); print("internet: OK")
    net = True
except Exception as e:
    print("internet: BLOCKED", repr(e)[:150]); net = False
if net:
    t = time.time()
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime"], capture_output=True)  # CPU build shadows the GPU one
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", "onnxruntime-gpu"], capture_output=True, text=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rtmlib"], capture_output=True)
    print("pip rtmlib + onnxruntime-gpu:", "ok" if r.returncode == 0 else r.stderr[-600:], f"({time.time()-t:.0f}s)")
    try:
        import torch  # loads CUDA/cuDNN libs that onnxruntime-gpu reuses
        import onnxruntime as ort
        try: ort.preload_dlls()
        except Exception as e: print("preload_dlls:", e)
        print("onnxruntime", ort.__version__, ort.get_available_providers())
        import numpy as np
        from rtmlib import Body
        dev = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        body = Body(mode="balanced", backend="onnxruntime", device=dev)
        img = (np.random.rand(480, 854, 3) * 255).astype("uint8"); body(img)
        t = time.time(); [body(img) for _ in range(30)]
        print(f"rtmlib on {dev}: {30/(time.time()-t):.1f} fps (random frames)")
    except Exception as e: print("rtmlib test failed:", repr(e)[:400])

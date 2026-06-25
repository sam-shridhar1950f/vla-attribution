import pathlib

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang", "cmake", "pkg-config", "python3-dev", "linux-libc-dev")
    .run_commands(
        "pip install uv",
        "git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi /openpi",
        "cd /openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync",
    )
    .env({"OPENPI_DATA_HOME": "/cache/openpi", "HF_HOME": "/cache/hf"})
    .add_local_file("worker.py", "/root/worker.py")
)

app = modal.App("vla-attribution", image=image)
cache = modal.Volume.from_name("vla-attribution-cache", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/cache": cache}, timeout=3600)
def attribute(bundle: bytes) -> bytes:
    import subprocess

    pathlib.Path("/cache/obs.npz").write_bytes(bundle)
    cache.commit()
    subprocess.run(["/openpi/.venv/bin/python", "/root/worker.py", "/cache/obs.npz", "/cache/out.npz"], check=True)
    cache.commit()
    return pathlib.Path("/cache/out.npz").read_bytes()


@app.local_entrypoint()
def main(bundle: str):
    src = pathlib.Path(bundle)
    out = src.with_name(src.stem + "_attribution.npz")
    out.write_bytes(attribute.remote(src.read_bytes()))
    print(f"wrote {out}")

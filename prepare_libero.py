import argparse
import io
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "physical-intelligence/libero"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"


def task_table():
    path = hf_hub_download(REPO, "meta/tasks.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    return {row["task_index"]: row["task"] for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    tasks = task_table()
    parquet = hf_hub_download(REPO, f"data/chunk-000/episode_{args.episode:06d}.parquet", repo_type="dataset")
    table = pq.read_table(parquet, columns=["image", "wrist_image", "state", "actions", "task_index"])
    n = min(args.frames, table.num_rows)

    def decode(column, i):
        return np.array(Image.open(io.BytesIO(table.column(column)[i].as_py()["bytes"])))

    exterior = np.stack([decode("image", i) for i in range(n)]).astype(np.uint8)
    wrist = np.stack([decode("wrist_image", i) for i in range(n)]).astype(np.uint8)
    state = np.array([table.column("state")[i].as_py() for i in range(n)], np.float32)
    actions = np.array([table.column("actions")[i].as_py() for i in range(n)], np.float32)
    prompt = tasks[table.column("task_index")[0].as_py()]
    alternatives = [t for t in tasks.values() if t != prompt][:3]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, dataset="libero", config_name="pi05_libero", checkpoint=CHECKPOINT,
        ext=exterior, wrist=wrist, proprio=state, actions_gt=actions,
        prompt=prompt, alt_prompts=alternatives,
    )
    print(f"{prompt!r}: {n} frames -> {args.out}")


if __name__ == "__main__":
    main()

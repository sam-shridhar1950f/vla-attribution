import argparse
import pathlib

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "lerobot/droid_100"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid"
EXTERIOR = "videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4"
WRIST = "videos/observation.images.wrist_image_left/chunk-000/file-000.mp4"


def read_frames(path, indices):
    capture = cv2.VideoCapture(path)
    wanted, frames, i = set(indices), {}, 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            wanted.discard(i)
        i += 1
    capture.release()
    return np.stack([frames[i] for i in indices]).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    task_table = pq.read_table(hf_hub_download(REPO, "meta/tasks.parquet", repo_type="dataset"))
    tasks = {
        task_table.column("task_index")[i].as_py(): task_table.column("__index_level_0__")[i].as_py()
        for i in range(task_table.num_rows)
    }
    table = pq.read_table(hf_hub_download(REPO, "data/chunk-000/file-000.parquet", repo_type="dataset"))
    episode = np.array(table.column("episode_index").to_pylist())
    start = int(np.where(episode == args.episode)[0].min())
    indices = list(range(start, start + min(args.frames, int((episode == args.episode).sum()))))

    state = np.array([table.column("observation.state")[i].as_py() for i in indices], np.float32)
    actions = np.array([table.column("action")[i].as_py() for i in indices], np.float32)
    prompt = tasks[table.column("task_index")[start].as_py()]
    alternatives = [t for t in tasks.values() if t.strip() and t != prompt][:3]

    exterior = read_frames(hf_hub_download(REPO, EXTERIOR, repo_type="dataset"), indices)
    wrist = read_frames(hf_hub_download(REPO, WRIST, repo_type="dataset"), indices)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, dataset="droid", config_name="pi05_droid", checkpoint=CHECKPOINT,
        ext=exterior, wrist=wrist, proprio=state, gripper=state[:, 6:7], actions_gt=actions,
        prompt=prompt, alt_prompts=alternatives,
    )
    print(f"{prompt!r}: {len(indices)} frames -> {args.out}")


if __name__ == "__main__":
    main()

import sys

import numpy as np
import openpi.training.config as config_lib
from openpi.policies import policy_config

NUM_NOISE = 6
MAX_FRAMES = 12
GIBBERISH = "qwp zxlk mfhd vbn"

CONDITIONS = {
    "drop_wrist": {"drop_wrist": True},
    "drop_exterior": {"drop_exterior": True},
    "drop_both_cams": {"drop_wrist": True, "drop_exterior": True},
    "zero_proprio": {"zero_proprio": True},
    "rand_proprio": {"rand_proprio": True},
    "lang_empty": {"prompt": ""},
    "lang_gibberish": {"prompt": GIBBERISH},
    "lang_swap": "swap",
}


def observation(bundle, t, rand_proprio_vec, *, prompt=None, drop_wrist=False,
                drop_exterior=False, zero_proprio=False, rand_proprio=False):
    proprio = bundle["proprio"][t].astype(np.float32)
    if zero_proprio:
        proprio = np.zeros_like(proprio)
    elif rand_proprio:
        proprio = rand_proprio_vec
    exterior = np.zeros_like(bundle["ext"][t]) if drop_exterior else bundle["ext"][t]
    wrist = np.zeros_like(bundle["wrist"][t]) if drop_wrist else bundle["wrist"][t]
    text = str(bundle["prompt"]) if prompt is None else prompt

    if str(bundle["dataset"]) == "libero":
        return {
            "observation/state": proprio,
            "observation/image": exterior,
            "observation/wrist_image": wrist,
            "prompt": text,
        }

    gripper = bundle["gripper"][t].astype(np.float32)
    if zero_proprio:
        gripper = np.zeros_like(gripper)
    elif rand_proprio:
        gripper = rand_proprio_vec[:1]
    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": proprio,
        "observation/gripper_position": gripper,
        "prompt": text,
    }


def run(obs_path, out_path):
    bundle = np.load(obs_path, allow_pickle=True)
    config = config_lib.get_config(str(bundle["config_name"]))
    policy = policy_config.create_trained_policy(config, str(bundle["checkpoint"]))
    horizon, action_dim = config.model.action_horizon, config.model.action_dim

    noise = np.random.default_rng(0).standard_normal((NUM_NOISE, horizon, action_dim)).astype(np.float32)
    rand_proprio_vec = np.random.default_rng(7).normal(0, 5, bundle["proprio"].shape[1]).astype(np.float32)
    alt_prompts = list(bundle["alt_prompts"])

    n = len(bundle["ext"])
    frames = list(range(0, n, max(1, n // MAX_FRAMES)))[:MAX_FRAMES]

    def predict(t, z, **kwargs):
        obs = observation(bundle, t, rand_proprio_vec, **kwargs)
        return np.asarray(policy.infer(obs, noise=z)["actions"], dtype=np.float64)

    floors, effects, per_dim = [], {c: [] for c in CONDITIONS}, {c: [] for c in CONDITIONS}
    for t in frames:
        base = np.stack([predict(t, z) for z in noise])
        floors.append(np.sqrt(((base - base.mean(0)) ** 2).mean()))
        for cond, spec in CONDITIONS.items():
            if spec == "swap":
                changed = np.stack([np.stack([predict(t, z, prompt=p) for z in noise]) for p in alt_prompts]).mean(0)
            else:
                changed = np.stack([predict(t, z, **spec) for z in noise])
            delta = changed - base
            effects[cond].append(float(np.sqrt((delta ** 2).mean())))
            per_dim[cond].append(np.sqrt((delta ** 2).mean(axis=(0, 1))))
        print(f"frame {t}", flush=True)

    np.savez(
        out_path,
        dataset=str(bundle["dataset"]),
        prompt=str(bundle["prompt"]),
        frames=np.array(frames),
        floors=np.array(floors),
        effects={c: np.array(effects[c]) for c in CONDITIONS},
        per_dim={c: np.array(per_dim[c]) for c in CONDITIONS},
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])

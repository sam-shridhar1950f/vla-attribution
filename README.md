# vla-attribution

A small probe that measures which inputs a VLA policy uses to decide its next action.

VLA policies take several inputs, like camera
views, the robot's proprioceptive state, and a language instruction. From these, it emits an action. Which of those inputs the policy *relies on* is not visible from the loss curve or the benchmark score. This tool investigates that, on any
[openpi](https://github.com/Physical-Intelligence/openpi) checkpoint, in a few
minutes on a single GPU.

## Method

An action is a chunk of numbers. π0.5 predicts several future timesteps of robot
command at once (for these checkpoints, a 10×7 grid). The probe measures which
inputs the policy uses to produce that grid:

1. **Baseline.** Run the policy on a real observation and record the action grid it
   predicts. The action head is a stochastic flow-matching sampler, so the same
   observation is run against a fixed bank of noise draws.
2. **Break one input.** Re-run with a single input corrupted, on the *same* noise
   draws so nothing else changes:
   - a camera replaced with a black image,
   - the proprioceptive state set to zeros or to a random vector,
   - the instruction removed, swapped for a different real task, or replaced with
     gibberish.
3. **Measure the change.** Take the root-mean-square difference between the
   perturbed action grid and the baseline grid (how far the predicted numbers
   moved), then divide it by how far the grid moves on its own when only the noise
   draw changes. That denominator is the policy's **sampling-noise floor**.

So every number is a multiple of the policy's own randomness:

- **about 1**: breaking the input moved the action no more than re-sampling the
  head would, so the policy barely uses it.
- **large** (for example 57): the action moved far beyond the policy's own noise,
  so the policy leans on that input heavily.
- **0**: the action did not change at all, so the input is ignored.

## Experiments on π0.5

Run on Physical Intelligence's two released π0.5 checkpoints, with frames
from each one's matching dataset (LIBERO is a sim; DROID is real
teleop data), with two scenes each.

![libero vs droid](results/figures/libero_vs_droid.png)

Each row is an input that was broken; each column is a policy. A cell is how far
that policy's action moved as a result, as a multiple of the policy's own sampling
noise. **Higher means it relies on that input more; about 1 or below means it
barely mattered; 0 means ignored.**

| broken input | pi05_libero (sim) | pi05_droid (real) |
| --- | ---: | ---: |
| drop both cameras | 60× | 3.2× |
| drop wrist camera | 57× | 1.7× |
| drop exterior camera | 8.0× | 1.8× |
| random proprioception | 0.00× | 3.5× ¹ |
| zero proprioception | 0.00× | 1.9× ¹ |
| remove instruction | 2.9× | 2.3× |
| swap instruction | 2.7× | 0.75× |
| gibberish instruction | 2.4× | 2.3× |

¹ DROID's released state is end-effector pose, not the joint angles pi05_droid
expects, so its proprioception numbers are suggestive rather than exact.

The two policies depend on seemingly
different components:

- **pi05_libero indexes on its wrist-camera.** The
  wrist view carries almost all of the signal (dropping it ≈ dropping both
  cameras), the exterior view adds little.
- **pi05_droid spreads its reliance.** No single input dominates; it uses
  proprioception and degrades gracefully when one camera is removed.
- **Both policies barely condition on instruction.** Swapping the task for a different real one moves the action about as much as sampling noise.

### The behavior is in the weights, not the input distribution

Feeding each policy the *other* domain's frames keeps each observed behavior too:

![cross distribution](results/figures/cross_distribution.png)

| | LIBERO in-sim | LIBERO on DROID | DROID in-real | DROID on LIBERO |
| --- | ---: | ---: | ---: | ---: |
| drop wrist | 57× | 13× | 1.7× | 0.6× |
| drop exterior | 8× | 3.2× | 1.8× | 0.9× |
| zero proprioception | 0.00× | 0.00× | 1.9× | 1.6× |
| random proprioception | 0.00× | 0.00× | 3.5× | 1.4× |
| sampling-noise floor | 0.023 | 0.137 | 0.075 | 0.162 |

The first four rows read as before (multiples of sampling noise). The last row is
the raw noise floor itself, the denominator the others are divided by, where a
higher value means the policy's actions are more erratic on those inputs.

On real out-of-distribution frames, pi05_libero still over-weights the wrist. Its state-blindness is
structural, not a property of clean sim images. With no fallback, it just becomes
erratic (its sampling-noise floor rises 6×). pi05_droid does the opposite: on
unfamiliar sim frames it leans *more* on proprioception and trusts the cameras
less, which is in practice graceful degradation under a distribution shift.

![sample frames](results/figures/sample_frames.png)

## Why this is useful

A benchmark score tells you a policy works. It does not tell you what the policy is
using to work, and that is what decides whether the score holds up outside the
benchmark. A policy can score well by leaning on one easy signal and ignoring the
rest. pi05_libero does this: it rides the wrist camera (57×) and is blind to its own
arm state (0). That is fine on a clean sim benchmark, but in the real world, where
that camera gets noisy or blocked, it has nothing to fall back on. The score hides
that brittleness, and this probe shows it.

It also catches inputs the policy is not using at all. A reading of 0 means the
input is either data being collected for nothing or a channel the model never
learned to use, both of which are worth knowing. And because it shows which sensors
actually drive the policy, it tells you where better data, labels, or sensor quality
will pay off, and where they will not.

## Setup

Local environment (data prep and plotting):

```bash
pip install -r requirements.txt
```

Inference runs on a cloud GPU via [Modal](https://modal.com).

```bash
modal setup
```

## Usage

```bash
# build an observation bundle from real frames
python prepare_libero.py --episode 0 --out data/libero_ep0.npz
python prepare_droid.py  --episode 0 --out data/droid_ep0.npz

# run the attribution sweep on a GPU
modal run probe.py --bundle data/libero_ep0.npz   # -> data/libero_ep0_attribution.npz

# regenerate the figures from results/runs
python figures.py
```

To attribute a different policy, point a bundle's `config_name` and `checkpoint`
at any openpi config and checkpoint.


## Layout

```
probe.py            Modal app: run the sweep on a GPU
worker.py           attribution logic (runs in the openpi environment)
prepare_libero.py   build a LIBERO observation bundle from HuggingFace
prepare_droid.py    build a DROID observation bundle from HuggingFace
figures.py          regenerate the figures from results/runs
results/runs/       precomputed attribution outputs
results/figures/    generated figures
```

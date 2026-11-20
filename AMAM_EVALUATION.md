# AMAM evaluation snapshot

The `amam-eval` branch freezes the environment code used by AMAM, rather than following the latest upstream release. The 10 task implementations and success predicates match the evaluated source snapshot. Adaptations add a websocket policy entry point, head-camera recordings and a Warp PCH concurrency fix. Model weights and training code are not included.

Install the dependencies declared in `pyproject.toml`; obtain simulator assets using the upstream asset instructions. `python main.py --help` describes the standalone websocket client. AMAM's paired VLM/VLA controller supplies subgoal/steering conditions and the benchmark normalization key; it is supplied by AMAM, not this environment repository.

AMAM evaluation uses 100 rollouts per task, an explicit seed list/rank partition, absolute joint actions, predicted chunks of 50 and execution of 25 actions per replan. Preserve the released task horizons and feasibility checks. Do not silently replace this branch with newer task code when reproducing a reported result.

`amam_environment.json` records source hashes. No checkpoints, rollout videos, private cluster launchers or credentials are distributed here.

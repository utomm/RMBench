"""Per-rank RMBench client that drives sims against an EVA-VLA inference server.

Adapted from the RoboTwin reference client for the RMBench (memory-dependent
manipulation) benchmark. The RMBench env API is identical to RoboTwin 2.0
(setup_demo / play_once / get_obs / take_action / check_success / close_env),
so the rollout logic is unchanged; the differences are:

  * the task set is RMBench's memory-dependent tasks (auto-discovered from
    ``envs/``), and ``--args.task-names all`` evaluates every one of them;
  * ``demo_clean`` here uses LargeView cameras + ``third_view: true``.

Launch one rank per GPU. Example (all tasks, 100 episodes each):

    python main.py --args.task-names all --args.episodes-per-task 100 \\
        --args.host 127.0.0.1 --args.port 9000 \\
        --args.experiment-root /path/work_dir \\
        --args.video-out-path /path/work_dir/videos
"""

from __future__ import annotations

import collections
import dataclasses
import importlib
import logging
import os
import pathlib
import sys
from datetime import datetime

import imageio
import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "policy"))
sys.path.insert(0, str(_REPO_ROOT / "description" / "utils"))

from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402


def discover_tasks() -> list[str]:
    """Every evaluatable RMBench task = an ``envs/<name>.py`` module (minus the
    private ``_`` files and subpackages). Sorted for deterministic rank splits.
    """
    env_dir = _REPO_ROOT / "envs"
    names = [
        p.stem
        for p in env_dir.glob("*.py")
        if not p.stem.startswith("_") and p.stem != "__init__"
    ]
    return sorted(names)


class WebsocketClientPolicy:
    """Thin wrapper over ``openpi_client.WebsocketClientPolicy`` (the same client
    the working ``policy/openpi_server`` uses), so the wire format matches the
    openpi inference server exactly.

    Lazy connect: with ``connect=False`` the underlying socket (and its
    background recv thread) is not opened until the first ``infer()`` call.
    Opening the websocket before the sim's curobo planner subprocesses are
    spawned (mp start method "spawn") can deadlock the curobo IPC, so we connect
    only once we're inside the rollout, after setup_demo has brought the
    planners up.
    """

    def __init__(self, host: str, port: int, connect: bool = True) -> None:
        self._host = host
        self._port = int(port)
        self._impl = None
        if connect:
            self._connect()

    def _connect(self) -> None:
        from openpi_client import websocket_client_policy as _wcp
        self._impl = _wcp.WebsocketClientPolicy(host=self._host, port=self._port)
        try:
            logging.info(f"[client] connected to ws://{self._host}:{self._port}; "
                         f"server metadata: {self._impl.get_server_metadata()}")
        except Exception:
            logging.info(f"[client] connected to ws://{self._host}:{self._port}")

    def infer(self, obs: dict) -> dict:
        if self._impl is None:
            self._connect()
        return self._impl.infer(obs)


def _build_task_args(task_name: str, task_config: str, ckpt_setting: str) -> dict:
    with open(_REPO_ROOT / "task_config" / f"{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        _embodiment_types = yaml.safe_load(f)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.safe_load(f)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    embodiment_type = args["embodiment"]
    if len(embodiment_type) == 1:
        rf = _embodiment_types[embodiment_type[0]]["file_path"]
        args["left_robot_file"] = rf
        args["right_robot_file"] = rf
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _embodiment_types[embodiment_type[0]]["file_path"]
        args["right_robot_file"] = _embodiment_types[embodiment_type[1]]["file_path"]
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment must have 1 or 3 entries")

    with open(os.path.join(args["left_robot_file"], "config.yml"), "r", encoding="utf-8") as f:
        args["left_embodiment_config"] = yaml.safe_load(f)
    with open(os.path.join(args["right_robot_file"], "config.yml"), "r", encoding="utf-8") as f:
        args["right_embodiment_config"] = yaml.safe_load(f)
    return args


def _build_obs_payload(observation: dict, instruction: str, norm_key: str | None = None) -> dict:
    # Matches policy/openpi_server/model.py's observation_window: the server-side
    # FormatRMBench transform reads state + images{cam_high,cam_left_wrist,
    # cam_right_wrist} + prompt. norm_key is included only if provided (ignored
    # by servers that don't use it).
    payload = {
        "state": np.ascontiguousarray(np.array(observation["joint_action"]["vector"], dtype=np.float32)),
        "images": {
            "cam_high": np.ascontiguousarray(np.array(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8)),
            "cam_left_wrist": np.ascontiguousarray(np.array(observation["observation"]["left_camera"]["rgb"], dtype=np.uint8)),
            "cam_right_wrist": np.ascontiguousarray(np.array(observation["observation"]["right_camera"]["rgb"], dtype=np.uint8)),
        },
        "prompt": str(instruction),
    }
    if norm_key is not None:
        payload["norm_key"] = norm_key
    return payload


def _run_episode(
    task_env, args: dict, client: WebsocketClientPolicy,
    seed: int, instruction_type: str, test_num: int,
    replan_steps: int, video_path: pathlib.Path | None,
    video_sample_every: int,
    norm_key: str | None = None,
    gripper_threshold: float | None = None,
) -> tuple[bool, int, int, str]:
    """Run one episode at `seed`. Returns (success, used_seed, advanced_seed, instruction)."""
    now_seed = seed
    while True:
        render_freq = args.get("render_freq", 0)
        args["render_freq"] = 0
        # expert planner first to validate seed
        try:
            task_env.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **args)
            episode_info = task_env.play_once()
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
            now_seed += 1
            args["render_freq"] = render_freq
            continue
        except Exception as exc:
            task_env.close_env()
            now_seed += 1
            args["render_freq"] = render_freq
            logging.warning(f"[expert] seed={now_seed - 1} {type(exc).__name__}: {exc}")
            continue

        if not (task_env.plan_success and task_env.check_success()):
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq
        # re-setup for the policy rollout from the same seed
        task_env.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **args)
        results = generate_episode_descriptions(args["task_name"], [episode_info["info"]], test_num)
        instruction = str(np.random.choice(results[0][instruction_type]))
        task_env.set_instruction(instruction=instruction)

        replay_images: list[np.ndarray] = []
        action_plan: collections.deque = collections.deque()
        succ = False
        rollout_error: Exception | None = None
        step_idx = 0
        # video_sample_every: 0 = sample only on policy-inference steps (free,
        # no extra get_obs); N>=1 = also force a video frame every N sim steps.
        sample_every = int(video_sample_every)
        try:
            while task_env.take_action_cnt < task_env.step_lim:
                # Skip get_obs() on sim steps where neither the policy nor the
                # video logger needs it. get_obs() does 3 GPU camera renders so
                # this is the dominant rollout-loop cost when not skipped.
                need_policy = not action_plan
                if sample_every <= 0:
                    want_video = video_path is not None and need_policy
                else:
                    want_video = video_path is not None and (step_idx % sample_every == 0)

                if need_policy or want_video:
                    obs = task_env.get_obs()
                    if want_video:
                        replay_images.append(np.ascontiguousarray(
                            np.array(obs["observation"]["head_camera"]["rgb"], dtype=np.uint8)))

                if need_policy:
                    response = client.infer(_build_obs_payload(obs, instruction, norm_key=norm_key))
                    actions = response.get("actions")
                    if actions is None:
                        raise RuntimeError(f"no actions in response: {sorted(response.keys())}")
                    action_chunk = np.asarray(actions)
                    if action_chunk.ndim == 3:
                        action_chunk = action_chunk[0]
                    take = action_chunk.shape[0] if replan_steps <= 0 else min(replan_steps, action_chunk.shape[0])
                    action_plan.extend(action_chunk[:take])

                # msgpack returns read-only views; copy so take_action gets a writable buffer
                action = np.array(action_plan.popleft(), dtype=np.float32, copy=True)
                if gripper_threshold is not None:
                    for gi in (6, 13):
                        action[gi] = 1.0 if action[gi] >= gripper_threshold else 0.0
                task_env.take_action(action.tolist())

                step_idx += 1
                if task_env.eval_success:
                    succ = True
                    if video_path is not None:
                        replay_images.append(np.ascontiguousarray(
                            np.array(task_env.now_obs["observation"]["head_camera"]["rgb"], dtype=np.uint8)))
                    break
        except Exception as exc:
            rollout_error = exc
        finally:
            task_env.test_num = getattr(task_env, "test_num", 0) + 1
            clear_cache_freq = max(int(args.get("clear_cache_freq", 5) or 1), 1)
            task_env.close_env(clear_cache=(task_env.test_num % clear_cache_freq == 0))
            if task_env.render_freq:
                task_env.viewer.close()

        if rollout_error is not None:
            logging.warning(
                f"[rollout] aborted step={len(replay_images)}/{task_env.step_lim} "
                f"seed={now_seed} {type(rollout_error).__name__}: {rollout_error}"
            )

        if video_path is not None and len(replay_images) >= 1:
            status = "success" if succ else "failed"
            video_path = video_path.with_name(f"{video_path.stem}_{status}{video_path.suffix}")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep playback rate roughly constant when frames are subsampled.
            video_fps = 2 if sample_every <= 0 else max(int(round(15 / sample_every)), 1)
            try:
                imageio.mimwrite(
                    str(video_path), replay_images, fps=video_fps,
                    codec="libx264", macro_block_size=1,
                    output_params=["-pix_fmt", "yuv420p"],
                )
            except Exception:
                logging.warning(f"failed to write video {video_path}")

        return succ, now_seed, now_seed + 1, instruction


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 9000

    # Comma-separated RMBench task list, or "all" to evaluate every task
    # discovered under envs/ (the default).
    task_names: str = "all"
    task_config: str = "demo_clean"
    ckpt_setting: str = "eva_vla"
    instruction_type: str = "unseen"

    episodes_per_task: int = 100
    replan_steps: int = -1
    seed: int = 0

    rank: int = 0
    # For multi-GPU launches: shard the task list round-robin across this many
    # ranks so each process evaluates a disjoint subset. 1 = this rank runs all
    # selected tasks.
    world_size: int = 1
    experiment_root: str = ""
    video_out_path: str = "eval_result"

    # Video frame cadence (sim steps).
    #   0 = sample only on policy-inference steps (no extra get_obs()).
    #   N>=1 = also force a frame every N sim steps.
    video_sample_every: int = 0

    # Norm key prefix for multi-dataset checkpoints. Combined with task_name
    # as "{norm_key_prefix}/{task_name}" and sent to the server. If empty,
    # derived from task_config by stripping the "demo_" prefix.
    norm_key_prefix: str = ""

    # Gripper binarization threshold. When set (not None), the gripper elements
    # of each action vector are snapped to 1.0 if >= threshold, else 0.0.
    gripper_threshold: float | None = None


_CSV_HEADER = "rank,task_name,episode_idx,success,seed,instruction\n"


def _resolve_tasks(args: Args) -> list[str]:
    """Expand ``--args.task-names`` (``all`` or a comma list) into the concrete
    task list this rank should run, applying round-robin sharding by rank.
    """
    raw = args.task_names.strip()
    if raw.lower() == "all":
        tasks = discover_tasks()
    else:
        requested = [t.strip() for t in raw.split(",") if t.strip()]
        available = set(discover_tasks())
        tasks = []
        for t in requested:
            if t not in available:
                raise RuntimeError(
                    f"unknown task '{t}'. available: {sorted(available)}"
                )
            tasks.append(t)
    if args.world_size > 1:
        tasks = [t for i, t in enumerate(tasks) if i % args.world_size == args.rank]
    return tasks


def _load_resume_state(csv_path: pathlib.Path) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Scan an existing per-rank CSV to plan resume.

    Returns:
        completed_eps: {task_name: count of rows seen}
        last_seed: {task_name: seed of the most recent row, used to advance st_seed}
        successes: {task_name: count of success=1 rows}
    """
    completed_eps: dict[str, int] = {}
    last_seed: dict[str, int] = {}
    successes: dict[str, int] = {}
    if not csv_path.exists():
        return completed_eps, last_seed, successes
    import csv as _csv  # local import keeps top-of-file clean

    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        # Old format CSVs don't have a "seed" column. Treat as full restart.
        if reader.fieldnames is None or "seed" not in reader.fieldnames:
            return {}, {}, {}
        for row in reader:
            t = row.get("task_name")
            if not t:  # truncated / malformed final row (e.g. killed mid-write)
                continue
            completed_eps[t] = completed_eps.get(t, 0) + 1
            try:
                last_seed[t] = int(row["seed"])
            except (TypeError, ValueError):
                pass
            try:
                successes[t] = successes.get(t, 0) + int(row["success"])
            except (TypeError, ValueError):
                pass
    return completed_eps, last_seed, successes


def eval_robotwin(args: Args) -> None:
    tasks = _resolve_tasks(args)
    if not tasks:
        logging.info(f"Rank {args.rank}: no tasks assigned (world_size={args.world_size}).")
        # Still emit the end marker so orchestrators waiting on all ranks proceed.
        output_root = (
            pathlib.Path(args.experiment_root) if args.experiment_root
            else pathlib.Path(args.video_out_path).parent
        )
        end_path = output_root / "end" / f"{args.rank}.end"
        end_path.parent.mkdir(parents=True, exist_ok=True)
        end_path.write_text("")
        return
    logging.info(f"Rank {args.rank} tasks ({len(tasks)}): {tasks}")

    np.random.seed(args.seed)
    output_root = pathlib.Path(args.experiment_root) if args.experiment_root else pathlib.Path(args.video_out_path).parent
    csv_path = output_root / "results" / f"{args.rank}.csv"
    end_path = output_root / "end" / f"{args.rank}.end"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    end_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: keep prior rows, only seed the file with a header on first run
    # or when the schema differs from the current header.
    completed_eps, last_seed_per_task, successes_per_task = _load_resume_state(csv_path)
    if not completed_eps:
        with open(csv_path, "w") as f:
            f.write(_CSV_HEADER)

    video_root = pathlib.Path(args.video_out_path) if args.video_out_path else None
    if video_root is not None:
        video_root.mkdir(parents=True, exist_ok=True)

    # Lazy connect: build the client now but defer the socket until the first
    # infer() call inside the rollout, so the websocket recv thread does not
    # coexist with curobo's "spawn" planner startup (which can deadlock).
    client = WebsocketClientPolicy(args.host, args.port, connect=False)

    total_episodes = sum(completed_eps.values())
    total_successes = sum(successes_per_task.values())
    started = datetime.now()

    norm_prefix = args.norm_key_prefix or args.task_config.removeprefix("demo_")

    for task_name in tasks:
        already_done = completed_eps.get(task_name, 0)
        if already_done >= args.episodes_per_task:
            logging.info(f"Skipping task={task_name} (already complete: {already_done} eps)")
            continue

        # Resume seed: continue from one past the last logged seed, otherwise
        # the same starting seed the first attempt used.
        if task_name in last_seed_per_task:
            st_seed = last_seed_per_task[task_name] + 1
        else:
            st_seed = 100000 * (1 + args.seed)

        norm_key = f"{norm_prefix}/{task_name}"
        logging.info(
            f"Starting task={task_name} rank={args.rank} seed_start={st_seed} "
            f"norm_key={norm_key} resume_from_ep={already_done}"
        )
        try:
            args_full = _build_task_args(task_name, args.task_config, args.ckpt_setting)
            args_full["eval_mode"] = True
            args_full["eval_video_log"] = False
            args_full["policy_name"] = "eva_vla_client"
            envs_module = importlib.import_module(f"envs.{task_name}")
            task_env = getattr(envs_module, task_name)()
        except Exception as exc:
            logging.error(f"Setup failed task={task_name} {type(exc).__name__}: {exc}")
            continue

        task_episodes = already_done
        task_successes = successes_per_task.get(task_name, 0)

        for ep_idx in range(already_done, args.episodes_per_task):
            video_path = video_root / f"{task_name}_ep{ep_idx}.mp4" if video_root is not None else None
            used_seed = st_seed
            try:
                success, used_seed, st_seed, instruction = _run_episode(
                    task_env=task_env, args=args_full, client=client,
                    seed=st_seed, instruction_type=args.instruction_type,
                    test_num=args.episodes_per_task, replan_steps=args.replan_steps,
                    video_path=video_path,
                    video_sample_every=args.video_sample_every,
                    norm_key=norm_key,
                    gripper_threshold=args.gripper_threshold,
                )
            except Exception as exc:
                logging.warning(f"[episode] task={task_name} ep={ep_idx} seed={st_seed} crashed {type(exc).__name__}: {exc}")
                success = False
                instruction = ""
                st_seed += 1

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            with open(csv_path, "a") as f:
                safe_instr = instruction.replace(",", " ").replace("\n", " ")
                f.write(f"{args.rank},{task_name},{ep_idx},{int(bool(success))},{used_seed},{safe_instr}\n")
            pct = task_successes / task_episodes * 100
            logging.info(
                f"[RESULT] {task_name}: Success rate: {task_successes}/{task_episodes} "
                f"=> {pct:.1f}%, current seed: {used_seed}"
            )

        if task_episodes:
            pct = task_successes / task_episodes * 100
            logging.info(f"Finished task={task_name} => {task_successes}/{task_episodes}={pct:.1f}%")

    end_path.write_text("")
    elapsed = (datetime.now() - started).total_seconds()
    if total_episodes:
        pct = total_successes / total_episodes * 100
        logging.info(f"Rank {args.rank} done. total={total_successes}/{total_episodes}={pct:.1f}%, elapsed={elapsed:.0f}s")
    else:
        logging.info(f"Rank {args.rank} done. no episodes ran.")


def _parse_args() -> "Args":
    """argparse front-end for Args (the RMBench .venv has no tyro).

    Accepts both ``--task-names`` and tyro-style ``--args.task-names`` so the
    existing eval.sh invocations keep working.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Per-rank RMBench websocket eval client")
    for f in dataclasses.fields(Args):
        typ = f.type
        default = f.default
        # Normalize the two accepted flag spellings to the same dest.
        dest = f.name
        flags = [f"--{f.name.replace('_', '-')}", f"--args.{f.name.replace('_', '-')}"]
        if typ is bool or default is True or default is False:
            parser.add_argument(*flags, dest=dest, type=lambda s: str(s).lower() in ("1", "true", "yes"),
                                default=default)
        elif f.name == "gripper_threshold":
            parser.add_argument(*flags, dest=dest,
                                type=lambda s: None if s in ("", "None", "none") else float(s),
                                default=default)
        elif isinstance(default, int) and not isinstance(default, bool):
            parser.add_argument(*flags, dest=dest, type=int, default=default)
        else:
            parser.add_argument(*flags, dest=dest, type=str, default=default)
    ns = parser.parse_args()
    return Args(**{f.name: getattr(ns, f.name) for f in dataclasses.fields(Args)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for name in ("curobo", "sapien", "mplib", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
    eval_robotwin(_parse_args())

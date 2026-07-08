"""Human-control playtest for defs with adaptive difficulty.

Drives the same config pipeline as training: loads an experiment def, applies
`adaptive_difficulty.ranges` at a difficulty d on top of the def's base world
preset, and runs human-controlled episodes via ratsim's run_human_session
(same R = next episode / Q = quit hotkeys).

Two modes:
  difficulty=<0..1>   fixed d — playtest one difficulty (0 = easiest, 1 = hardest)
  adaptive=1          d starts at the def's d0 and moves EXACTLY like training:
                      after each episode, success = pickups >= success_pickups
                      -> d += step_up, failure -> d -= step_down. Ending an
                      episode early with R uses the same rule on the pickups so
                      far (bailing out with too few = failure), matching how
                      training scores truncated episodes.

Usage (same key=value style as train.py; needs the Unity EDITOR on :9000 in play mode):
    python human_control_test_adaptive.py def=ortho_wells_adaptive difficulty=0.0
    python human_control_test_adaptive.py def=ortho_wells_adaptive difficulty=1.0
    python human_control_test_adaptive.py def=ortho_wells_adaptive adaptive=1
    python human_control_test_adaptive.py def=ortho_wells_adaptive adaptive=1 rtf=2.0 seed=7
"""
import sys
from pathlib import Path

from ratsim.roslike_unity_connector.connector import RoslikeUnityConnector
from ratsim.roslike_unity_connector.message_definitions import StringMessage
from ratsim.config_blender import blend_presets, to_entries_json
from ratsim.human_control_test import run_human_session

sys.path.insert(0, str(Path(__file__).parent))
from experiment_defs import load_experiment_def, resolve_def_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "ratsim_wildfire_gym_env"))
from ratsim_wildfire_gym_env.adaptive_difficulty import interpolate_ranges  # noqa: E402

DEFS_DIR = Path(__file__).parent / "defs"


def _parse_kv_args(argv):
    out = {}
    for a in argv:
        if "=" not in a:
            print(f"ERROR: expected key=value args, got '{a}'")
            sys.exit(1)
        k, v = a.split("=", 1)
        out[k] = v
    return out


def main():
    args = _parse_kv_args(sys.argv[1:])
    if "def" not in args:
        print("Usage: human_control_test_adaptive.py def=<rundef> "
              "[difficulty=<0..1> | adaptive=1] [rtf=1.0] [seed=<int>]")
        sys.exit(1)

    exp = load_experiment_def(resolve_def_path(DEFS_DIR, args.pop("def")))
    ad = exp.adaptive_difficulty
    if ad is None:
        print(f"ERROR: def '{exp.exp_id}' has no adaptive_difficulty block.")
        sys.exit(1)

    adaptive = args.pop("adaptive", "0") not in ("0", "false", "False", "")
    fixed_d = args.pop("difficulty", None)
    if adaptive and fixed_d is not None:
        print("ERROR: pass either difficulty= or adaptive=1, not both.")
        sys.exit(1)
    d = float(fixed_d) if fixed_d is not None else ad.d0
    if not 0.0 <= d <= 1.0:
        print(f"ERROR: difficulty must be in [0, 1], got {d}")
        sys.exit(1)

    rtf = float(args.pop("rtf", 1.0))
    seed = int(args.pop("seed")) if "seed" in args else None
    if args:
        print(f"ERROR: unknown args: {list(args)}")
        sys.exit(1)

    # Baseline variation's presets (same resolution train.py uses for it).
    base_world = blend_presets("world", list(exp.world_preset))
    agent_config = blend_presets("agents", list(exp.agent_preset))
    task_config = blend_presets("task", list(exp.task_preset))

    conn = RoslikeUnityConnector(verbose=False)
    conn.connect()
    conn.publish(StringMessage(data="Wildfire"), "/sim_control/scene_select")
    conn.send_messages_and_step(enable_physics_step=False)
    conn.read_messages_from_unity()
    conn.publish(StringMessage(data=to_entries_json(agent_config)), "/sim_control/agent_config")
    conn.send_messages_and_step(enable_physics_step=False)
    conn.read_messages_from_unity()

    mode = "ADAPTIVE (training-like walk)" if adaptive else f"FIXED d={d:.2f}"
    print(f"def={exp.exp_id}  mode={mode}  success_pickups={ad.success_pickups}"
          + (f"  step +{ad.step_up}/-{ad.step_down}" if adaptive else ""))

    episode = 0
    current_seed = seed
    try:
        while True:
            episode += 1
            overrides = interpolate_ranges(ad.ranges, d)
            world_config = {**base_world, **overrides}
            print(f"\n{'='*60}")
            print(f"Episode {episode} (seed={current_seed})  d={d:.3f}  ->  "
                  f"{ {k: overrides[k] for k in list(overrides)[:4]} } ...")

            result = run_human_session(conn, world_config, agent_config, task_config,
                                       seed=current_seed, rtf=rtf, max_steps=None)
            pickups = result["objects_found"]
            success = pickups >= ad.success_pickups
            print(f"\nEpisode over: pickups={pickups} -> "
                  f"{'SUCCESS' if success else 'FAILURE'} (need >= {ad.success_pickups})")

            if adaptive:
                d += ad.step_up if success else -ad.step_down
                d = min(1.0, max(0.0, d))
                print(f"difficulty -> {d:.3f}")

            if result.get("quit_requested"):
                break
            current_seed = 0 if current_seed is None else current_seed + 1
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()

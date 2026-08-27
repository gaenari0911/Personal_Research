#!/usr/bin/env python3
"""Restore LIBERO demonstration states and extract semantic predicate traces.

This utility intentionally uses the versions pinned by the upstream LIBERO
repository (robosuite 1.4.0 and MuJoCo 2.3.x).  It does not step the simulator:
each HDF5 state is restored independently and interpreted as the post-action
observation at the same index.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import robosuite

from libero.libero.envs.env_wrapper import ControlEnv


ROOT = Path(__file__).resolve().parents[1]
BDDL_ROOT = ROOT / "external/LIBERO/libero/libero/bddl_files/libero_10"
DATA_ROOT = Path("/ssd1/itaein/datasets/LIBERO/libero_10")
LIBERO_ASSETS = ROOT / "external/LIBERO/libero/libero/assets"
ROBOSUITE_ASSETS = Path(robosuite.__file__).resolve().parent / "models/assets"

TASKS = {
    3: {
        "bddl": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl",
        "hdf5": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
        "predicates": {
            "object_in_target": ["in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"],
            "articulation_closed": ["close", "white_cabinet_1_bottom_region"],
        },
        "objects": ["akita_black_bowl_1"],
        "articulation": "white_cabinet_1_bottom_region",
    },
    9: {
        "bddl": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl",
        "hdf5": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
        "predicates": {
            "object_in_target": ["in", "white_yellow_mug_1", "microwave_1_heating_region"],
            "articulation_closed": ["close", "microwave_1"],
        },
        "objects": ["white_yellow_mug_1"],
        "articulation": "microwave_1",
    },
    4: {
        "bddl": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl",
        "hdf5": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
        "predicates": {
            "white_mug_on_left_plate": ["on", "porcelain_mug_1", "plate_1"],
            "yellow_white_mug_on_right_plate": ["on", "white_yellow_mug_1", "plate_2"],
        },
        "objects": ["porcelain_mug_1", "white_yellow_mug_1"],
        "articulation": None,
    },
}


def fix_asset_paths(xml: str) -> str:
    return (
        xml.replace(
            "/home/yifengz/workspace/libero-dev/chiliocosm/assets",
            str(LIBERO_ASSETS),
        )
        .replace(
            "/home/yifengz/workspace/robosuite-master/robosuite/models/assets",
            str(ROBOSUITE_ASSETS),
        )
    )


def vector(values):
    return [round(float(x), 8) for x in np.asarray(values).reshape(-1)]


def extract_demo(env: ControlEnv, demo, config):
    xml = fix_asset_paths(demo.attrs["model_file"])
    env.reset_from_xml_string(xml)
    states = demo["states"]
    actions = demo["actions"][:]
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"state/action length mismatch: {states.shape} vs {actions.shape}")
    if env.get_sim_state().shape != states.shape[1:]:
        raise ValueError(
            f"flattened state mismatch: sim {env.get_sim_state().shape}, file {states.shape[1:]}"
        )

    base = env.env
    rows = []
    for t, state in enumerate(states):
        env.set_state(state)
        env.sim.forward()
        predicates = {
            name: bool(base._eval_predicate(spec))
            for name, spec in config["predicates"].items()
        }
        grasps = {
            obj: bool(
                base._check_grasp(
                    gripper=base.robots[0].gripper,
                    object_geoms=base.get_object(obj),
                )
            )
            for obj in config["objects"]
        }
        object_pos = {
            obj: vector(base.object_states_dict[obj].get_geom_state()["pos"])
            for obj in config["objects"]
        }
        articulation_qpos = None
        if config["articulation"]:
            art_state = base.object_states_dict[config["articulation"]]
            if art_state.object_state_type == "site":
                joints = base.object_sites_dict[config["articulation"]].joints
                articulation_qpos = vector(
                    [env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(j)] for j in joints]
                )
            else:
                articulation_qpos = vector(art_state.get_joint_state())
        gripper = base.robots[0].gripper
        gripper_qpos = []
        for joint in gripper.joints:
            addr = env.sim.model.get_joint_qpos_addr(joint)
            gripper_qpos.extend(np.asarray(env.sim.data.qpos[addr]).reshape(-1))
        rows.append(
            {
                "t": t,
                "predicates": predicates,
                "grasps": grasps,
                "object_pos": object_pos,
                "articulation_qpos": articulation_qpos,
                "eef_pos": vector(env.sim.data.site_xpos[base.robots[0].eef_site_id]),
                "gripper_qpos": vector(gripper_qpos),
                "action_gripper": round(float(actions[t, -1]), 8),
            }
        )
    return {
        "demo_id": demo.name.rsplit("/", 1)[-1],
        "T": len(rows),
        "state_dim": int(states.shape[1]),
        "rows": rows,
    }


def parse_demo_ids(text, available):
    if text == "all":
        return sorted(available, key=lambda x: int(x.rsplit("_", 1)[-1]))
    return [f"demo_{int(x)}" for x in text.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=TASKS, required=True)
    parser.add_argument("--demos", default="all", help="comma-separated numeric ids or all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = TASKS[args.task]
    env = ControlEnv(
        bddl_file_name=str(BDDL_ROOT / config["bddl"]),
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
    )
    env.reset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(DATA_ROOT / config["hdf5"], "r") as source, args.output.open(
            "w", encoding="utf-8"
        ) as sink:
            demos = parse_demo_ids(args.demos, source["data"].keys())
            for demo_id in demos:
                result = extract_demo(env, source["data"][demo_id], config)
                result.update(
                    {
                        "task_id": args.task,
                        "hdf5": config["hdf5"],
                        "bddl": config["bddl"],
                        "state_alignment": "action[t] -> env.step -> obs[t] (post-action)",
                    }
                )
                sink.write(json.dumps(result, separators=(",", ":")) + "\n")
                print(f"task {args.task} {demo_id}: {result['T']} states", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

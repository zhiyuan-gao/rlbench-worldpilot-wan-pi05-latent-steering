from __future__ import annotations

import argparse
from pathlib import Path
import unittest
from unittest import mock

from rlbench_worldpilot_wan_pi05 import eval_online_rlbench_rpc_block12_per_event_v1 as protocol
from rlbench_worldpilot_wan_pi05.eval_online_rlbench_rpc_block12_per_event_v1 import (
    advance_event,
    episode_uid,
    stable_seed,
    validate_frozen_protocol,
)


class Block12PerEventV1Test(unittest.TestCase):
    def test_episode_uid_and_seed_do_not_depend_on_row_order(self):
        row = {"task": "open_drawer", "variation": "variation1", "episode": "episode23"}
        uid = episode_uid(row)
        self.assertEqual(uid, "open_drawer/variation1/episode23")
        seed_a = stable_seed(eval_seed=0, uid=uid, stream="action", index=4)
        seed_b = stable_seed(eval_seed=0, uid=uid, stream="action", index=4)
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, stable_seed(eval_seed=1, uid=uid, stream="action", index=4))
        self.assertNotEqual(seed_a, stable_seed(eval_seed=0, uid=uid, stream="action", index=5))

    def test_middle_event_advances(self):
        self.assertEqual(
            advance_event(
                event_idx=0,
                num_events=3,
                event_step_idx=4,
                switch_reason="pose_reached",
                final_event_locked=False,
            ),
            (1, 0, False),
        )

    def test_last_event_locks_instead_of_exiting(self):
        self.assertEqual(
            advance_event(
                event_idx=2,
                num_events=3,
                event_step_idx=4,
                switch_reason="pose_reached",
                final_event_locked=False,
            ),
            (2, 0, True),
        )

    def test_locked_last_event_stays_locked(self):
        self.assertEqual(
            advance_event(
                event_idx=2,
                num_events=3,
                event_step_idx=5,
                switch_reason=None,
                final_event_locked=True,
            ),
            (2, 5, True),
        )

    def test_frozen_protocol_validation(self):
        args = argparse.Namespace(
            split="val",
            max_episodes_per_task=25,
            selection="first",
            wan_mode="matched",
            wan_seed_mode="per_event",
            wan_text_source="task",
            event_switch_mode="pose_or_steps",
            max_steps=30,
            max_steps_per_event=8,
            fixed_steps_per_event=None,
            event_goal_pos_threshold=0.04,
            event_goal_rot_threshold=0.5,
            event_goal_gripper_threshold=0.5,
            continue_on_invalid=False,
            seed=2,
        )
        validate_frozen_protocol(args)
        args.wan_seed_mode = "per_step"
        with self.assertRaises(ValueError):
            validate_frozen_protocol(args)

    def test_unsuccessful_final_event_continues_to_thirty_steps(self):
        observation = object()
        demo = argparse.Namespace(_observations=[observation, observation])
        task_env = mock.Mock()
        task_env.reset_to_demo.return_value = (None, observation)
        task_env.step.return_value = (observation, 0.0, False)
        args = argparse.Namespace(
            lowdim_root_200=Path("/low200"),
            lowdim_root_400=Path("/low400"),
            rgb_root_200=Path("/rgb200"),
            rgb_root_400=Path("/rgb400"),
            record_video=False,
            image_size=256,
            seed=0,
            wan_text_source="task",
            state_gripper_threshold=0.95,
            wan_mode="matched",
            wan_seed_mode="per_event",
            ignore_collision=True,
            continue_on_invalid=False,
            event_goal_pos_threshold=0.04,
            event_goal_rot_threshold=0.5,
            event_goal_gripper_threshold=0.5,
            max_steps_per_event=8,
            max_steps=30,
            record_failures_only=True,
            out=Path("/tmp/result.jsonl"),
            video_fps=2,
            event_switch_mode="pose_or_steps",
        )
        row = {
            "task": "open_drawer",
            "variation": "variation0",
            "episode": "episode0",
            "task_instruction": "open the drawer",
            "source_bundle": "local200",
            "num_frames": 2,
        }
        reach_metrics = {
            "event_goal_pos_error": 0.0,
            "event_goal_rotvec_error": 0.0,
            "event_goal_gripper_error": 0.0,
        }
        with (
            mock.patch.object(protocol._legacy, "root_for_source", return_value=Path("/root")),
            mock.patch.object(protocol, "resolve_episode_dir", return_value=Path("/episode")),
            mock.patch.object(protocol, "load_demo", return_value=demo),
            mock.patch.object(protocol, "event_boundaries", return_value=[0, 1]),
            mock.patch.object(protocol, "read_goal_images_from_episode", return_value={}),
            mock.patch.object(protocol, "obs_rgb_images", return_value={}),
            mock.patch.object(protocol, "policy_obs_from_rlbench", return_value={}),
            mock.patch.object(protocol._legacy, "infer_action7", return_value=[0.0] * 7),
            mock.patch.object(protocol, "action7_to_rlbench_action9", return_value=protocol.np.zeros(9)),
            mock.patch.object(protocol, "pose_event_reached", return_value=(True, reach_metrics)),
        ):
            result = protocol.run_episode_rpc(row_idx=17, row=row, task_env=task_env, args=args)

        self.assertFalse(result["success"])
        self.assertTrue(result["final_event_locked"])
        self.assertEqual(result["executed_steps"], 30)
        self.assertEqual(task_env.step.call_count, 30)
        self.assertEqual(len({step["wan_seed"] for step in result["step_results"]}), 1)
        self.assertEqual(len({step["action_seed"] for step in result["step_results"]}), 30)


if __name__ == "__main__":
    unittest.main()

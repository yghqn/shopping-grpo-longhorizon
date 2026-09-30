"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_grpo import build_command, main as train_grpo_main, parse_args
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(
            sys,
            "argv",
            [
                "shopping-grpo",
                "evaluate",
                str(root / "examples/trajectories.jsonl"),
            ],
        ), patch("builtins.print") as output:
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model.resolve()))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train.resolve()))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation.resolve()))
        self.assertEqual(
            environment["SHOPPING_GRPO_DIAGNOSTICS_PATH"],
            str(output.resolve() / "training_diagnostics.jsonl"),
        )
        self.assertIn("trainer.logger=[console]", command)

    def test_grpo_launcher_runs_preflight_with_hydra_overrides(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"example")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--experiment-name",
                    "smoke",
                    "--",
                    "trainer.total_training_steps=2",
                ],
            ), patch(
                "scripts.train_grpo.subprocess.call",
                side_effect=[0, 0],
            ) as subprocess_call:
                with self.assertRaises(SystemExit) as result:
                    train_grpo_main()

        self.assertEqual(result.exception.code, 0)
        self.assertEqual(subprocess_call.call_count, 2)
        preflight = subprocess_call.call_args_list[0].args[0]
        self.assertIn("trainer.logger=[console]", preflight)
        self.assertIn("trainer.experiment_name=smoke", preflight)
        self.assertIn("trainer.total_training_steps=2", preflight)

    def test_grpo_launcher_requires_explicit_resume_for_nonempty_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"example")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            output.mkdir()
            (output / "existing").write_text("data", encoding="utf-8")
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--dry-run",
                ],
            ):
                args = parse_args()
            with self.assertRaisesRegex(
                SystemExit,
                "output directory must be new or empty",
            ):
                build_command(args)

    def test_grpo_launcher_resumes_only_from_complete_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"example")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            actor = output / "global_step_100" / "actor"
            actor.mkdir(parents=True)
            (output / "latest_checkpointed_iteration.txt").write_text(
                "100",
                encoding="utf-8",
            )
            for name in (
                "model_world_size_1_rank_0.pt",
                "optim_world_size_1_rank_0.pt",
                "extra_state_world_size_1_rank_0.pt",
            ):
                (actor / name).write_bytes(b"example")
            (output / "global_step_100" / "data.pt").write_bytes(b"example")
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--resume",
                    "--dry-run",
                    "--",
                    "trainer.resume_mode=disable",
                ],
            ):
                args = parse_args()
            command, _ = build_command(args)

        self.assertEqual(command.count("trainer.resume_mode=auto"), 1)
        self.assertNotIn("trainer.resume_mode=disable", command)

    def test_grpo_config_declares_checkpoint_resume_controls(self):
        config = (
            Path(__file__).resolve().parents[1] / "configs/grpo.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("resume_mode: disable", config)
        self.assertIn("resume_from_path: null", config)
        self.assertIn("max_actor_ckpt_to_keep: null", config)
        self.assertIn("max_critic_ckpt_to_keep: null", config)


if __name__ == "__main__":
    unittest.main()

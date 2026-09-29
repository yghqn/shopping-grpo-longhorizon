import json
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def task_ids(relative_path):
    path = ROOT / relative_path
    return {
        int(json.loads(line)["task_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


class BudgetMemoryHoldoutTest(unittest.TestCase):
    def test_metadata_hashes_and_task_ids_match_frozen_files(self):
        metadata = json.loads(
            (ROOT / "data/budget_memory/metadata.json").read_text(encoding="utf-8")
        )
        for split in ("smoke", "holdout"):
            record = metadata[split]
            path = ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            self.assertEqual(task_ids(record["path"]), set(record["task_ids"]))
            self.assertEqual(len(record["task_ids"]), record["count"])

    def test_smoke_and_holdout_partition_grpo_validation(self):
        smoke = task_ids("data/budget_memory/dev_smoke.jsonl")
        holdout = task_ids("data/budget_memory/dev_holdout_40.jsonl")
        validation = task_ids("data/grpo/validation.jsonl")

        self.assertEqual(len(smoke), 10)
        self.assertEqual(len(holdout), 40)
        self.assertFalse(smoke & holdout)
        self.assertEqual(smoke | holdout, validation)

    def test_holdout_is_disjoint_from_training_and_final_evaluation(self):
        holdout = task_ids("data/budget_memory/dev_holdout_40.jsonl")
        training = set().union(
            task_ids("data/grpo/train.jsonl"),
            task_ids("data/sft/train.jsonl"),
            task_ids("data/sft/validation.jsonl"),
        )
        final_evaluation = task_ids("data/evaluation/tasks.jsonl")

        self.assertFalse(holdout & training)
        self.assertFalse(holdout & final_evaluation)


if __name__ == "__main__":
    unittest.main()

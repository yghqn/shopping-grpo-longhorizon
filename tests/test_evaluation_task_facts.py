import unittest

from shopping_grpo.evaluation.rubric import extract_rubric_candidates
from shopping_grpo.evaluation.task_facts import (
    task_facts_from_product_corpus,
)


class ProductCorpusTaskFactsTests(unittest.TestCase):
    def setUp(self):
        self.products = [
            {
                "asin": "10001",
                "category": "家居›枕头›乳胶枕",
                "title": "护颈乳胶枕",
                "brand": "示例牌",
                "shop_name": "示例店",
                "pricing": [99.0],
                "attribute": ["护颈"],
                "customization_options": {"颜色": ["白色"]},
                "instructions": [
                    {
                        "instruction": "找一个100元以内的护颈乳胶枕",
                        "attributes": ["护颈"],
                        "instruction_options": [],
                    }
                ],
            }
        ]

    def test_builds_facts_and_candidates_in_requested_order(self):
        rows = task_facts_from_product_corpus(
            task_ids=[0],
            products=self.products,
            reward_goal_builder=lambda instruction, product: {
                "category": product["category"],
                "expected_core_functions": instruction["attributes"],
                "price_upper": 100.0,
                "reward_feature_version": "test-v1",
            },
        )
        self.assertEqual(rows[0]["task_id"], 0)
        self.assertEqual(rows[0]["query"], "找一个100元以内的护颈乳胶枕")
        candidates = extract_rubric_candidates(rows[0])
        self.assertEqual(candidates["task_id"], 0)
        self.assertGreaterEqual(len(candidates["candidates"]), 3)

    def test_rejects_ambiguous_instruction_selection(self):
        products = [dict(self.products[0])]
        products[0]["instructions"] = [
            self.products[0]["instructions"][0],
            self.products[0]["instructions"][0],
        ]
        with self.assertRaisesRegex(ValueError, "exactly one instruction"):
            task_facts_from_product_corpus(task_ids=[0], products=products)

    def test_rejects_duplicate_task_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            task_facts_from_product_corpus(
                task_ids=[0, 0],
                products=self.products,
            )


if __name__ == "__main__":
    unittest.main()

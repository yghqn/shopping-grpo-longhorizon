import unittest

from shopping_grpo.training.grpo.memory_runtime import (
    initialize_candidate_memory,
    update_candidate_memory,
)


def search_state():
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results",
        "search_available": False,
        "actions": ["back to search", "35842622441"],
        "query": "乳胶枕",
        "normalized_query": "乳胶枕",
        "page": 1,
        "total_pages": 1,
        "total_results": 1,
        "rank_start": 1,
        "rank_end": 1,
        "products": [
            {
                "rank": 1,
                "asin": "35842622441",
                "title": "泰国乳胶枕",
                "brand": "示例品牌",
                "category": "枕头",
                "price": 20,
                "key_attributes": ["高低款"],
            }
        ],
    }


class GrpoMemoryRuntimeTest(unittest.TestCase):
    def test_renders_replaceable_memory_without_actionable_historical_asin(self):
        memory, initial = initialize_candidate_memory(
            requirement="预算30元的乳胶枕",
            initial_observation_state=None,
            token_budget=2048,
            count_tokens=len,
        )

        content, view = update_candidate_memory(
            memory,
            observation_state=search_state(),
            observation=(
                "[SHOPPING_OBSERVATION_V2]\npage_type: search_results\n"
                "1|35842622441|20.0|示例品牌|枕头|高低款|泰国乳胶枕\n"
                "搜索功能是否可用: False\n可点击的按钮: [\"35842622441\"]"
            ),
            step_index=0,
            tool_name="search_products",
            token_budget=2048,
            count_tokens=len,
        )

        self.assertEqual(initial["selected_count"], 0)
        self.assertIn("[SHOPPING_TRAINING_MEMORY_CONTEXT_V1]", content)
        self.assertIn("[END_SHOPPING_TRAINING_MEMORY_CONTEXT_V1]", content)
        self.assertIn("当前可打开商品 ASIN: 35842622441", content)
        memory_section = content.split("[SHOPPING_CANDIDATE_MEMORY_V1]", 1)[1]
        self.assertNotIn("35842622441", memory_section)
        self.assertEqual(view["selected_count"], 1)
        self.assertNotIn("content", view)
        self.assertIn("content_sha256", view)


if __name__ == "__main__":
    unittest.main()

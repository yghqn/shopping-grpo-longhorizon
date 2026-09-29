import unittest

from shopping_grpo.environment.observation import StructuredObservationError
from shopping_grpo.memory import CandidateMemory, MemoryBudgetError


def search_state(price=20):
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results",
        "search_available": False,
        "actions": ["back to search", "12345678"],
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
                "asin": "12345678",
                "title": "泰国乳胶枕",
                "brand": "示例品牌",
                "category": "枕头",
                "price": price,
                "key_attributes": ["高低款"],
            }
        ],
    }


def detail_state(price=25):
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "product_detail",
        "search_available": False,
        "actions": ["back to search", "Buy Now", "高款"],
        "product": {
            "asin": "12345678",
            "title": "泰国乳胶枕",
            "brand": "示例品牌",
            "category": "枕头",
            "price": 20,
            "key_attributes": ["高低款"],
        },
        "selected_price": price,
        "selected_options": {"高度": "高款"},
        "available_options": {"高度": ["低款", "高款"]},
    }


class CandidateMemoryTest(unittest.TestCase):
    def test_search_observation_creates_candidate_with_provenance(self):
        memory = CandidateMemory("预算 30 元的乳胶枕")

        update = memory.observe(search_state(), step_index=0, tool_name="search_products")
        candidate = memory.snapshot()["candidates"][0]

        self.assertEqual(update.candidates_created, 1)
        self.assertEqual(candidate["asin"], "12345678")
        self.assertEqual(candidate["facts"]["price"][0]["value"], 20)
        self.assertEqual(
            candidate["facts"]["price"][0]["sources"][0],
            {
                "step_index": 0,
                "tool_name": "search_products",
                "page_type": "search_results",
            },
        )

    def test_detail_merges_options_and_preserves_conflicting_prices(self):
        memory = CandidateMemory("预算 30 元的乳胶枕")
        memory.observe(search_state(price=20), step_index=0, tool_name="search_products")

        memory.observe(detail_state(price=25), step_index=1, tool_name="open_product")
        facts = memory.snapshot()["candidates"][0]["facts"]

        self.assertEqual([item["value"] for item in facts["price"]], [20, 25])
        self.assertEqual(facts["selected_option.高度"][0]["value"], "高款")
        self.assertEqual(facts["available_option.高度"][0]["value"], ["低款", "高款"])

    def test_same_value_adds_source_without_duplicating_fact(self):
        memory = CandidateMemory("乳胶枕")
        memory.observe(search_state(), step_index=0, tool_name="search_products")
        memory.observe(search_state(), step_index=2, tool_name="search_products")

        price = memory.snapshot()["candidates"][0]["facts"]["price"]

        self.assertEqual(len(price), 1)
        self.assertEqual([source["step_index"] for source in price[0]["sources"]], [0, 2])

    def test_private_fields_are_rejected_recursively(self):
        memory = CandidateMemory("乳胶枕")
        state = detail_state()
        state["product"]["reward_detail"] = {"r_price": 1}

        with self.assertRaisesRegex(StructuredObservationError, "forbidden"):
            memory.observe(state, step_index=1, tool_name="open_product")

    def test_render_contains_evidence_but_no_action_list(self):
        memory = CandidateMemory("预算 30 元的乳胶枕")
        memory.observe(search_state(), step_index=0, tool_name="search_products")

        rendered = memory.render()

        self.assertIn("候选 12345678", rendered)
        self.assertIn("s0:search_products:search_results", rendered)
        self.assertIn("可执行动作必须以最新 observation 为准", rendered)
        self.assertNotIn("Buy Now", rendered)
        self.assertNotIn("back to search", rendered)

    def test_budget_prefers_verified_candidate_and_reports_dropped_items(self):
        memory = CandidateMemory("预算 30 元的乳胶枕")
        state = search_state()
        second = dict(state["products"][0])
        second.update({"rank": 2, "asin": "87654321", "title": "已核验乳胶枕"})
        state["products"].append(second)
        state["actions"].append("87654321")
        state["total_results"] = 2
        state["rank_end"] = 2
        memory.observe(state, step_index=0, tool_name="search_products")
        detail = detail_state()
        detail["product"]["asin"] = "87654321"
        memory.observe(detail, step_index=1, tool_name="open_product")

        selected_one = None
        for budget in range(1, len(memory.render()) + 1):
            try:
                view = memory.render_view(token_budget=budget, count_tokens=len)
            except MemoryBudgetError:
                continue
            if len(view.selected_asins) == 1:
                selected_one = view
                break

        self.assertIsNotNone(selected_one)
        self.assertEqual(selected_one.selected_asins, ("87654321",))
        self.assertEqual(selected_one.dropped_asins, ("12345678",))
        self.assertLessEqual(selected_one.used_tokens, selected_one.token_budget)

    def test_budget_smaller_than_fixed_header_fails(self):
        memory = CandidateMemory("乳胶枕")

        with self.assertRaisesRegex(MemoryBudgetError, "header"):
            memory.render_view(token_budget=1, count_tokens=len)

    def test_budget_selection_uses_logarithmic_tokenizer_calls(self):
        memory = CandidateMemory("乳胶枕")
        state = search_state()
        products = []
        actions = ["back to search"]
        for rank in range(1, 21):
            product = dict(state["products"][0])
            product.update({"rank": rank, "asin": f"{rank:08d}"})
            products.append(product)
            actions.append(product["asin"])
        state.update(
            {
                "products": products,
                "actions": actions,
                "total_results": 20,
                "rank_end": 20,
            }
        )
        memory.observe(state, step_index=0, tool_name="search_products")
        calls = 0

        def counter(text):
            nonlocal calls
            calls += 1
            return len(text)

        memory.render_view(token_budget=len(memory.render()), count_tokens=counter)

        self.assertLessEqual(calls, 7)


if __name__ == "__main__":
    unittest.main()

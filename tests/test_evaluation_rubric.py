import unittest

from shopping_grpo.evaluation.rubric import (
    extract_price_candidates,
    extract_rubric_candidates,
    materialize_rubric_bundle,
)


class EvaluationRubricTests(unittest.TestCase):
    def test_price_extraction_rejects_dimensions_and_accepts_yuan_budget(self):
        query = (
            "直径约50毫米，可用于套设直径50至70毫米的线缆，"
            "预算控制30元左右。"
        )

        result = extract_price_candidates(query)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["constraint_type"], "price_preference")
        self.assertEqual(result[0]["expected_value"]["value"], 30)

    def test_price_extraction_accepts_kuai_currency(self):
        result = extract_price_candidates("预算在200块左右")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["constraint_type"], "price_preference")
        self.assertEqual(result[0]["expected_value"]["value"], 200)

    def test_metric_option_covers_chinese_unit_spans_without_false_price(self):
        query = (
            "材质为304不锈钢，直径约50毫米，可用于套设直径50至70毫米"
            "的线缆，预算控制30元左右。"
        )
        facts = {
            "schema_version": "shopping-task-facts-v1",
            "task_id": 15576,
            "query": query,
            "task_data_hash": "task-hash",
            "query_hash": "query-hash",
            "instruction": {"instruction_options": []},
            "target_product": {
                "category": "金属材料及制品›金属丝/绳/缆",
                "attributes": ["304不锈钢"],
                "customization_options": {
                    "规格": [{"value": "50mm可套50-70mm直径"}]
                },
            },
            "reward_features": {
                "expected_model": ["304"],
                "expected_core_functions": ["304不锈钢"],
                "required_options_by_key": {
                    "size": {
                        "source_axis": "规格",
                        "value": "50mm可套50-70mm直径",
                    }
                },
            },
        }

        result = extract_rubric_candidates(facts)
        price_types = {
            item["constraint_type"]
            for item in result["candidates"]
            if item["field_path"] == "purchase.price"
        }
        option = next(
            item
            for item in result["candidates"]
            if item["constraint_type"] == "option"
        )
        phrases = {
            item["expected_value"]
            for item in result["candidates"]
            if item["constraint_type"] == "query_phrase"
        }

        self.assertEqual(price_types, {"price_preference"})
        self.assertEqual(
            {span["text"] for span in option["query_spans"]},
            {"50毫米", "50至70毫米", "70毫米"},
        )
        self.assertNotIn("直径约", phrases)

    def test_query_backstop_exposes_requirements_missing_from_reward_features(self):
        query = (
            "求一对卡通-永结同心款的高档酒红色木梳，希望配备礼盒，"
            "能作为新娘结婚的陪嫁物品，价格在20元左右。"
        )
        facts = {
            "schema_version": "shopping-task-facts-v1",
            "task_id": 8187,
            "query": query,
            "task_data_hash": "task-hash",
            "query_hash": "query-hash",
            "instruction": {
                "attributes": ["高档", "结婚", "陪嫁", "卡通-永结同心"],
                "instruction_options": ["【卡通-永结同心】2个装"],
            },
            "target_product": {
                "category": "节庆用品/礼品›婚庆用品›新娘配件",
                "title": "结婚梳子一对红色木梳礼盒装",
                "attributes": ["高档", "结婚", "陪嫁", "卡通-永结同心"],
                "customization_options": {
                    "颜色分类": [
                        {"value": "【卡通-永结同心】2个装"},
                        {"value": "礼盒装【酒红纯色梳】2个装"},
                    ]
                },
            },
            "reward_features": {
                "expected_core_functions": [
                    "高档",
                    "结婚",
                    "陪嫁",
                    "卡通-永结同心",
                ],
                "required_options_by_key": {
                    "color": {
                        "source_axis": "颜色分类",
                        "value": "【卡通-永结同心】2个装",
                    }
                },
            },
        }

        result = extract_rubric_candidates(facts)
        by_value = {
            item["expected_value"]: item
            for item in result["candidates"]
            if isinstance(item["expected_value"], str)
        }

        self.assertIn("酒红色木梳", by_value)
        self.assertIn("礼盒", by_value)
        self.assertEqual(
            [span["text"] for span in by_value["【卡通-永结同心】2个装"]["query_spans"]],
            ["一对", "卡通-永结同心"],
        )

    def test_specific_option_shadows_duplicate_core_function(self):
        query = "求一对卡通-永结同心款木梳"
        facts = {
            "schema_version": "shopping-task-facts-v1",
            "task_id": 1,
            "query": query,
            "task_data_hash": "task-hash",
            "query_hash": "query-hash",
        }
        candidates = {
            "schema_version": "shopping-rubric-candidates-v1",
            "extractor_version": "shopping-rubric-extractor-v6",
            "task_id": 1,
            "query": query,
            "task_data_hash": "task-hash",
            "query_hash": "query-hash",
            "candidates": [
                {
                    "candidate_id": "c0001",
                    "constraint_type": "core_function",
                    "field_path": "product.attributes",
                    "operator": "contains",
                    "expected_value": "卡通-永结同心",
                    "hardness_source": "explicit_core_function_rule",
                    "query_spans": [
                        {"start": 3, "end": 10, "text": "卡通-永结同心"}
                    ],
                    "data_sources": ["query"],
                },
                {
                    "candidate_id": "c0002",
                    "constraint_type": "option",
                    "field_path": "purchase.options.color",
                    "operator": "eq",
                    "expected_value": "【卡通-永结同心】2个装",
                    "hardness_source": "explicit_option_rule",
                    "query_spans": [],
                    "data_sources": ["query"],
                },
            ],
        }
        response = {
            "selected_constraints": [
                {
                    "candidate_id": "c0001",
                    "description": "应为卡通-永结同心款",
                    "hardness": "hard",
                    "query_quote": "卡通-永结同心",
                    "selection_reason": "Query 明确要求",
                },
                {
                    "candidate_id": "c0002",
                    "description": "应选择卡通-永结同心两件装",
                    "hardness": "hard",
                    "query_quote": "一对卡通-永结同心款",
                    "selection_reason": "具体选项承载该要求",
                },
            ]
        }

        bundle = materialize_rubric_bundle(
            task_facts=facts,
            candidates=candidates,
            curator_response=response,
            curator_model="deepseek-flash",
            curator_prompt_version="rubric-curator-v1-draft-r6",
            rubric_version="test-rubric-v1",
        )

        self.assertEqual(
            [item["candidate_id"] for item in bundle["rubrics"]],
            ["c0002"],
        )

    def test_specific_option_shadows_duplicate_query_phrase(self):
        query = "采用打孔式安装"
        facts = {
            "schema_version": "shopping-task-facts-v1",
            "task_id": 2,
            "query": query,
            "task_data_hash": "task-hash",
            "query_hash": "query-hash",
        }
        candidates = {
            "schema_version": "shopping-rubric-candidates-v1",
            "extractor_version": "shopping-rubric-extractor-v6",
            "task_id": 2,
            "query": query,
            "task_data_hash": "task-hash",
            "query_hash": "query-hash",
            "candidates": [
                {
                    "candidate_id": "c0001",
                    "constraint_type": "query_phrase",
                    "field_path": "product.visible_evidence",
                    "operator": "semantically_supports",
                    "expected_value": "采用打孔式安装",
                    "hardness_source": "explicit_query_phrase_rule",
                    "query_spans": [],
                    "data_sources": ["query"],
                },
                {
                    "candidate_id": "c0002",
                    "constraint_type": "option",
                    "field_path": "purchase.options.size",
                    "operator": "eq",
                    "expected_value": "宽1.3*高2.7【打孔款一片】",
                    "hardness_source": "explicit_option_rule",
                    "query_spans": [],
                    "data_sources": ["query"],
                },
            ],
        }
        response = {
            "selected_constraints": [
                {
                    "candidate_id": "c0001",
                    "description": "采用打孔安装",
                    "hardness": "hard",
                    "query_quote": query,
                    "selection_reason": "Query 明确要求",
                },
                {
                    "candidate_id": "c0002",
                    "description": "选择打孔款",
                    "hardness": "hard",
                    "query_quote": query,
                    "selection_reason": "具体选项承载要求",
                },
            ]
        }

        bundle = materialize_rubric_bundle(
            task_facts=facts,
            candidates=candidates,
            curator_response=response,
            curator_model="deepseek-flash",
            curator_prompt_version="rubric-curator-v1-draft-r6",
            rubric_version="test-rubric-v1",
        )

        self.assertEqual(
            [item["candidate_id"] for item in bundle["rubrics"]],
            ["c0002"],
        )


if __name__ == "__main__":
    unittest.main()

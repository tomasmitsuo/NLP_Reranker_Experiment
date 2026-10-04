"""Testa os módulos do experimento com corpus pequeno, sem downloads ou modelos remotos.

Execução: .venv/bin/python -m unittest discover -s tests -v
"""
import copy
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi


def load_experiment_functions():
    from qexp import config, retrieval, selection, generation, runner, evaluation
    namespace = {}
    for module in (config, retrieval, selection, generation, runner, evaluation):
        namespace.update(vars(module))
    return namespace


class FakeGenerator:
    def __init__(self, response=None):
        self.calls = []
        self.response = response

    def __call__(self, prompt, max_new_tokens, seed):
        self.calls.append(prompt)
        if self.response is not None:
            return self.response
        if "JSON array" in prompt:
            return json.dumps(["myocardial infarction therapy", "cardiac muscle injury",
                               "coronary ischemia treatment", "cardiovascular clinical intervention"])
        return "Myocardial infarction and coronary ischemia involve cardiac muscle injury and cardiovascular therapy."


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_experiment_functions()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = copy.deepcopy(self.ns["QEXP_CONFIG"])
        self.config.update(cache_dir=self.temp.name, candidate_n=4, eval_k=2, signal_top_k=3,
                           gap_threshold=0.0, entropy_threshold=1.0, overlap_threshold=0.0,
                           divergence_threshold=1.0)
        self.documents = {
            "d1": "heart disease cardiac therapy",
            "d2": "myocardial infarction coronary ischemia",
            "d3": "cardiac muscle injury therapy",
            "d4": "heart anatomy structural biology",
            "d5": "gene protein expression cellular signaling",
            "d6": "protein gene mutation expression",
            "d7": "forest plant ecology photosynthesis",
            "d8": "planet orbit astronomy telescope",
            "d9": "nutrition metabolism diet glucose",
            "d10": "quantum physics photon laser",
            "d11": "volcano earthquake ocean geology",
            "d12": "mathematics algebra polynomial topology",
        }
        tokens = self.ns["tokenize"]
        self.case = dict(config={"name": "Toy", "slug": "toy"}, documents=self.documents,
                         doc_ids=list(self.documents), queries={"q1": "heart", "q2": "gene"},
                         qrels={"q1": {"d2": 2, "d1": 1}, "q2": {"d5": 1}},
                         bm25=BM25Okapi([tokens(text) for text in self.documents.values()]))
        self.rerank_calls = []

    def rerank(self, query, candidates, documents):
        self.rerank_calls.append((query, [item["doc_id"] for item in candidates]))
        # Ordem deliberadamente diferente para exercitar divergência e reutilização de BGE.
        return [dict(item, score=float(i)) for i, item in enumerate(reversed(candidates))]

    def run_experiment(self, generator=None):
        generator = generator if generator is not None else FakeGenerator()
        cache = self.ns["QExpExpansionCache"](self.config, generator)
        result = self.ns["run_qexp_experiment"]({"toy": self.case}, self.config, cache, self.rerank)
        return result, generator, cache

    def test_rrf_votes_ranks_deduplication_and_cutoff(self):
        rankings = [[{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "a"}],
                    [{"doc_id": "b"}, {"doc_id": "c"}]]
        result = self.ns["qexp_rrf"](rankings, 60, 2)
        self.assertEqual([item["doc_id"] for item in result], ["b", "a"])
        self.assertAlmostEqual(result[0]["score"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(result[1]["score"], 1 / 61)

    def test_signals_no_qrels_and_uniform_or_empty_scores(self):
        candidates = [{"doc_id": "d1", "score": 10.0}, {"doc_id": "d2", "score": 10.0}]
        signals = self.ns["qexp_difficulty_signals"]("heart", candidates, self.documents, self.config)
        self.assertAlmostEqual(signals["score_entropy"], 1.0)
        self.assertEqual(signals["score_gap"], 0)
        self.assertEqual(signals["lexical_overlap"], 0.5)
        empty = self.ns["qexp_difficulty_signals"]("heart", [], self.documents, self.config)
        self.assertEqual(empty["signal_docs"], 0)
        peaked = self.ns["qexp_softmax"]([1e6, -1e6])
        np.testing.assert_equal(peaked, [1.0, 0.0])

    def test_rule_reasons_and_rank_divergence_detect_reordering(self):
        signals = dict(relative_gap=0.02, score_entropy=0.95, lexical_overlap=0.1)
        config = {**self.config, "gap_threshold": 0.15, "entropy_threshold": 0.9, "overlap_threshold": 0.3}
        decision, reason = self.ns["qexp_should_expand"](signals, config)
        self.assertTrue(decision)
        self.assertEqual(set(reason.split(";")), {"gap", "entropy", "overlap"})
        first = [{"doc_id": str(i)} for i in range(4)]
        divergence = self.ns["qexp_rank_divergence"](first, first[::-1], 4)
        self.assertGreater(divergence, 0)
        self.assertEqual(self.ns["qexp_rank_divergence"](first, first, 4), 0)

    def test_cache_reuses_generation_and_invalidates_changed_prompt_configuration(self):
        generator = FakeGenerator()
        cache = self.ns["QExpExpansionCache"](self.config, generator)
        first = cache.get("heart", "multi")
        second = cache.get("heart", "multi")
        self.assertEqual(len(generator.calls), 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["generation_sec"], second["generation_sec"])
        changed = self.ns["QExpExpansionCache"]({**self.config, "model_revision": "different"}, generator)
        third = changed.get("heart", "multi")
        self.assertNotEqual(first["cache_key"], third["cache_key"])
        self.assertEqual(len(generator.calls), 2)

    def test_malformed_generation_is_logged_not_cached(self):
        cache = self.ns["QExpExpansionCache"](self.config, FakeGenerator("not JSON"))
        record = cache.get("heart", "multi")
        self.assertEqual(record["status"], "invalid_generation")
        self.assertEqual(record["expansions"], [])
        self.assertTrue(record["error"])
        self.assertEqual(list(Path(self.temp.name).glob("*.json")), [])
        with self.assertRaises(ValueError):
            self.ns["qexp_parse_generation"]('["heart", "heart", "heart", "heart"]', "heart", "multi", 4)

    def test_rm3_normalization_original_weight_and_weighted_bm25(self):
        initial = self.ns["retrieve_bm25"]("heart", self.case["bm25"], self.case["doc_ids"], 4)
        weights = self.ns["qexp_rm3_weights"]("heart", initial, self.documents, self.config)
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertGreaterEqual(weights["heart"], self.config["rm3_original_weight"])
        self.assertTrue(set(weights) - {"heart"})
        pure = self.ns["qexp_rm3_weights"]("heart", initial, self.documents,
                                           {**self.config, "rm3_original_weight": 1.0})
        self.assertEqual(pure, {"heart": 1.0})
        original = self.ns["qexp_weighted_bm25"](pure, self.case["bm25"], self.case["doc_ids"], 4)
        self.assertEqual(original, initial)

    def test_easy_policies_reuse_B_without_generation_and_h4_keeps_candidates(self):
        result, generator, _ = self.run_experiment()
        self.assertEqual(len(generator.calls), 4)  # C e D, uma geração de cada tipo por query.
        metrics = result["metrics"].set_index(["query_id", "pipeline_id"])
        self.assertEqual(set(result["metrics"]["pipeline_id"]), {"A", "B", "C", "D", "E", "E_RA", "F", "C_H4"})
        for query_id in self.case["queries"]:
            baseline = metrics.loc[(query_id, "B")]
            for policy in ("E", "E_RA"):
                row = metrics.loc[(query_id, policy)]
                self.assertFalse(row["expansion_requested"])
                self.assertEqual(row["ndcg@2"], baseline["ndcg@2"])
                self.assertEqual(row["generation_sec"], 0)
                self.assertAlmostEqual(row["estimated_uncached_sec"],
                                       baseline["estimated_uncached_sec"] + row["decision_sec"])
            rankings = result["rankings"]
            subset = rankings[(rankings.query_id == query_id) & (rankings.stage == "candidates")]
            self.assertEqual(subset[subset.pipeline_id == "C"].doc_id.tolist(),
                             subset[subset.pipeline_id == "C_H4"].doc_id.tolist())
        for artifact in result["artifacts"]:
            if artifact["pipeline_id"] != "C_H4":
                self.assertEqual(artifact["reranker_query"], artifact["original_query"])
        sizes = result["rankings"].groupby(["query_id", "pipeline_id", "stage"]).size()
        self.assertLessEqual(sizes.max(), self.config["candidate_n"])

    def test_hard_policies_match_D_and_charge_extra_initial_BGE(self):
        self.config.update(entropy_threshold=0.0)
        result, _, _ = self.run_experiment()
        metrics = result["metrics"].set_index(["query_id", "pipeline_id"])
        for query_id in self.case["queries"]:
            for policy in ("E", "E_RA"):
                row = metrics.loc[(query_id, policy)]
                self.assertTrue(row["expansion_applied"])
                self.assertEqual(row["ndcg@2"], metrics.loc[(query_id, "D"), "ndcg@2"])
            ra = metrics.loc[(query_id, "E_RA")]
            self.assertGreater(ra["initial_rerank_sec"], 0)
            self.assertGreater(ra["rerank_sec"], 0)
            self.assertAlmostEqual(ra["estimated_uncached_sec"], sum(ra[name] for name in
                                   ["retrieval_sec", "decision_sec", "generation_sec", "rerank_sec", "initial_rerank_sec"]))

    def test_qrels_changes_only_evaluation(self):
        before, _, _ = self.run_experiment()
        self.case["qrels"] = {"q1": {"d12": 1}, "q2": {"d12": 1}}
        after, _, _ = self.run_experiment()
        pd.testing.assert_frame_equal(before["decisions"], after["decisions"])
        pd.testing.assert_frame_equal(before["rankings"].drop(columns="relevance"),
                                      after["rankings"].drop(columns="relevance"))

    def test_export_hypotheses_pairs_and_missing_variation(self):
        result, _, _ = self.run_experiment()
        output = Path(self.temp.name) / "outputs"
        self.ns["qexp_export"](result, self.config, output)
        self.assertEqual(set(result["hypotheses"]["hypothesis"]), {"H1", "H2", "H3", "H4"})
        self.assertTrue((result["hypotheses"].num_pairs == 2).all())
        self.assertTrue((result["hypotheses"].p_value_holm >= result["hypotheses"].p_value).all())
        self.assertTrue((result["h2_associations"].status == "insufficient_variation").all())
        self.assertTrue((output / "qexp_report.md").exists())
        self.assertEqual(json.loads((output / "qexp_metadata.json").read_text())["config"]["candidate_n"], 4)
        broken = pd.concat([result["metrics"], result["metrics"].iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            self.ns["qexp_hypothesis_tables"](broken, self.config)

    def test_failed_expansions_fall_back_and_candidate_recall_uses_each_ranking(self):
        self.config.update(entropy_threshold=0.0)
        result, _, _ = self.run_experiment(FakeGenerator(""))
        metrics = result["metrics"].set_index(["query_id", "pipeline_id"])
        for query_id in self.case["queries"]:
            base = metrics.loc[(query_id, "B")]
            for policy in ("C", "D", "E", "E_RA", "C_H4"):
                row = metrics.loc[(query_id, policy)]
                self.assertTrue(row["expansion_requested"])
                self.assertFalse(row["expansion_applied"])
                self.assertEqual(row["generation_status"], "invalid_generation")
                self.assertEqual(row["ndcg@2"], base["ndcg@2"])
        for (query_id, pipeline_id), row in metrics.iterrows():
            ranking = result["rankings"]
            candidates = ranking[(ranking.query_id == query_id) &
                                 (ranking.pipeline_id == pipeline_id) & (ranking.stage == "candidates")]
            expected = sum(candidates.relevance > 0) / len(self.case["qrels"][query_id])
            self.assertEqual(row["candidate_recall@4"], expected)

    def test_generator_infrastructure_error_is_not_silenced(self):
        def unavailable(*args):
            raise RuntimeError("model unavailable")
        cache = self.ns["QExpExpansionCache"](self.config, unavailable)
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            cache.get("heart", "multi")

    def test_invalid_config_rejected_before_retrieval(self):
        for change in ({"candidate_n": 1}, {"max_queries_per_dataset": 0}, {"num_reformulations": 2},
                       {"rm3_original_weight": 1.1}, {"selector_rule": "qrels"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.ns["qexp_validate_config"]({**self.config, **change})


if __name__ == "__main__":
    unittest.main()

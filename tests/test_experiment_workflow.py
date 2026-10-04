"""Integração offline: retomada, invalidação, etapas e leitura do estudo anterior."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import test_query_expansion as fixtures
from qexp.checkpoints import run_with_checkpoints, save_result
from qexp.cli import execute_stage
from qexp.generation import QExpExpansionCache
from pln_legacy.io import load_baseline_results


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        # Reusa somente o corpus de teste, sem herdar/reexecutar a suíte inteira.
        if not hasattr(fixtures.ExpansionTests, "ns"):
            fixtures.ExpansionTests.setUpClass()
        self.fixture = fixtures.ExpansionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture.config
        self.experiments = {"toy": self.fixture.case}
        self.root = Path(self.fixture.temp.name)
        self.progress = patch("qexp.runner.tqdm", lambda value, **kwargs: value)
        self.progress.start()
        self.addCleanup(self.progress.stop)

    def run_checkpointed(self, rerank=None, resume=True):
        return run_with_checkpoints(
            self.experiments, self.config, QExpExpansionCache(self.config, fixtures.FakeGenerator()),
            rerank or self.fixture.rerank, self.root / "checkpoints", resume=resume,
        )

    def test_resume_does_not_call_models_and_preserves_results(self):
        original = self.run_checkpointed()
        def fail(*args):
            raise AssertionError("Model must not run on resume")
        resumed = self.run_checkpointed(fail)
        for name in ("metrics", "rankings", "decisions", "summary"):
            pd.testing.assert_frame_equal(original[name], resumed[name], check_exact=False, atol=1e-14, rtol=1e-14)
        self.assertEqual(resumed["metadata"]["checkpoint_queries_reused"], 2)
        self.assertEqual(resumed["metadata"]["checkpoint_queries_executed"], 0)

    def test_interruption_keeps_completed_query(self):
        def interrupt(query, candidates, documents):
            if query == "gene":
                raise RuntimeError("interrupted")
            return self.fixture.rerank(query, candidates, documents)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_checkpointed(interrupt)
        self.fixture.rerank_calls.clear()
        result = self.run_checkpointed()
        self.assertEqual(result["metadata"]["checkpoint_queries_reused"], 1)
        self.assertTrue(all(query.startswith("gene") for query, _ in self.fixture.rerank_calls))

    def test_checkpoint_invalidation_for_config_corpus_and_qrels(self):
        self.run_checkpointed()
        self.config["gap_threshold"] = 0.1
        self.assertEqual(self.run_checkpointed()["metadata"]["checkpoint_queries_executed"], 2)
        self.fixture.case["qrels"]["q1"] = {"d12": 1}
        self.assertEqual(self.run_checkpointed()["metadata"]["checkpoint_queries_executed"], 1)
        self.fixture.case["documents"]["d1"] += " edited"
        self.assertEqual(self.run_checkpointed()["metadata"]["checkpoint_queries_executed"], 2)
        self.assertEqual(self.run_checkpointed(resume=False)["metadata"]["checkpoint_queries_executed"], 2)

    def test_analyze_without_preparing_data_or_loading_models(self):
        result = self.run_checkpointed()
        save_result(result, self.root / "qexp_run.json")
        with patch("qexp.datasets.prepare_dataset", side_effect=AssertionError("No datasets")), \
             patch("qexp.generation.QExpLocalGenerator.ensure_loaded", side_effect=AssertionError("No LLM")), \
             patch("qexp.reranking.BGEReranker.ensure_loaded", side_effect=AssertionError("No BGE")):
            analyzed = execute_stage("analyze", {"output_dir": str(self.root)})
        self.assertEqual(set(analyzed["hypotheses"].hypothesis), {"H1", "H2", "H3", "H4"})
        self.assertTrue((self.root / "qexp_report.md").exists())

    def test_shared_multi_query_ranking_runs_only_once(self):
        self.config["entropy_threshold"] = 0
        self.run_checkpointed()
        # B, C, C_H4, D/E/E_RA compartilhado e F: cinco chamadas por query.
        self.assertEqual(len(self.fixture.rerank_calls), 10)

    def test_prepared_cache_distinguishes_smoke_and_full(self):
        from qexp.datasets import prepare_dataset
        case = self.fixture.case
        calls = []
        def fake_loader(config, max_queries=None):
            calls.append(max_queries)
            return case["documents"], case["queries"], case["qrels"]
        with patch("qexp.datasets.load_beir_dataset", new=fake_loader):
            prepare_dataset(case["config"], self.root / "prepared", 3)
            prepare_dataset(case["config"], self.root / "prepared", 3)
            prepare_dataset(case["config"], self.root / "prepared", None)
        self.assertEqual(calls, [3, None])

    def test_checkpoint_preserves_numeric_looking_query_ids(self):
        case = self.fixture.case
        case["queries"] = {"001": "heart"}
        case["qrels"] = {"001": {"d1": 1}}
        self.run_checkpointed()
        resumed = self.run_checkpointed()
        self.assertEqual(set(resumed["metrics"].query_id), {"001"})

    def test_legacy_reader_preserves_ids_and_rejects_unpaired_results(self):
        rankings = pd.DataFrame([dict(dataset="Toy", query_id="001", pipeline="BM25", doc_text="text")])
        metrics = pd.DataFrame([dict(dataset="Toy", query_id="001", pipeline=p)
                                for p in ("BM25", "BM25 + BGE-Reranker")])
        rankings.to_csv(self.root / "rankings_all_datasets.csv", index=False)
        metrics.to_csv(self.root / "results_all_datasets.csv", index=False)
        loaded, _ = load_baseline_results(self.root)
        self.assertEqual(loaded.query_id.iloc[0], "001")
        metrics.loc[1, "query_id"] = "002"
        metrics.to_csv(self.root / "results_all_datasets.csv", index=False)
        with self.assertRaisesRegex(ValueError, "mesmas queries"):
            load_baseline_results(self.root)


if __name__ == "__main__":
    unittest.main()

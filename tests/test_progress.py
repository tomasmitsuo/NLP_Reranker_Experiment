"""Progresso real em stream, retomada e estado parcial em falhas; sem downloads."""
import io
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rich.console import Console
from qexp.progress import ProgressReporter, report, activity, active
from qexp.checkpoints import run_with_checkpoints
from qexp.generation import QExpExpansionCache
import test_query_expansion as fixtures


class ProgressTests(unittest.TestCase):
    def setUp(self):
        if not hasattr(fixtures.ExpansionTests, "ns"):
            fixtures.ExpansionTests.setUpClass()
        self.fixture = fixtures.ExpansionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = Path(self.fixture.temp.name)
        self.stream = io.StringIO()
        self.console = Console(file=self.stream, force_terminal=False, color_system=None)

    def reporter(self, mode="plain"):
        return ProgressReporter(self.root, mode, console=self.console)

    def run_queries(self, rerank=None):
        return run_with_checkpoints({"toy": self.fixture.case}, self.fixture.config,
                                    QExpExpansionCache(self.fixture.config, fixtures.FakeGenerator()),
                                    rerank or self.fixture.rerank, self.root / "checkpoints")

    def test_plain_stream_reports_counts_and_resume(self):
        with self.reporter():
            self.run_queries()
        state = json.loads((self.root / "qexp_progress.json").read_text())
        self.assertEqual(state["completed_queries"], 2)
        self.assertEqual(state["total_queries"], 2)
        self.assertEqual(state["executed_queries"], 2)
        self.assertEqual(state["status"], "completed")
        self.assertIn("queries 1/2", self.stream.getvalue())
        self.assertIn("B: reranqueamento BGE original", self.stream.getvalue())
        self.assertNotIn("\x1b", self.stream.getvalue())
        with self.reporter():
            self.run_queries()
        state = json.loads((self.root / "qexp_progress.json").read_text())
        self.assertEqual(state["reused_queries"], 2)
        self.assertEqual(state["executed_queries"], 0)
        self.assertFalse(active())

    def test_failure_keeps_partial_progress(self):
        def fail(query, candidates, documents):
            if query == "gene":
                raise RuntimeError("simulated error")
            return self.fixture.rerank(query, candidates, documents)
        with self.assertRaises(RuntimeError), self.reporter():
            self.run_queries(fail)
        state = json.loads((self.root / "qexp_progress.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["completed_queries"], 1)
        self.assertEqual(state["query_id"], "q2")
        self.assertIn("simulated error", state["error"])
        self.assertFalse(active())

    def test_heartbeat_does_not_increment_progress_and_loading_restores_stage(self):
        with self.reporter() as tracker:
            report("BGE", total_queries=3)
            with activity("Download"):
                tracker.update(heartbeat=True)
                self.assertEqual(tracker.state["completed_queries"], 0)
            self.assertEqual(tracker.state["phase"], "BGE")
        self.assertIn("sem avanço confirmado", self.stream.getvalue())

    def test_off_has_no_side_effects(self):
        with self.reporter("off"), contextlib.redirect_stderr(self.stream):
            report("ignored", complete=True)
            self.run_queries()
        self.assertFalse((self.root / "qexp_progress.json").exists())
        self.assertEqual(self.stream.getvalue(), "")

    def test_interactive_display_uses_rich_and_closes(self):
        with patch.dict("os.environ", {"TERM": "xterm-256color"}):
            console = Console(file=self.stream, force_terminal=True, width=100)
            with ProgressReporter(self.root, console=console) as tracker:
                report("Executando", total_queries=2)
                report(complete=True)
                self.assertIsNotNone(tracker.display)
                self.assertEqual(tracker.display.tasks[0].completed, 1)
                self.assertEqual(tracker.display.tasks[0].description, "Executando")
                report(complete=True)
        self.assertIn("Concluído", self.stream.getvalue())

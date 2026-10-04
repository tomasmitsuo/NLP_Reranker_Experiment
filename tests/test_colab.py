"""Testes offline da integração: nenhuma sessão ou unidade Colab é utilizada."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qexp.cli import load_settings
from qexp.colab import build_bundle, execute_remote, launcher_source
from qexp.remote_worker import collect_results, restore_state, run_worker

PROJECT = Path(__file__).resolve().parents[1]


class ColabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"project_root": ".", "datasets": ["scifact"],
                                          "output_dir": "out", "qexp": {"max_queries_per_dataset": 3}}))
        self.commands = []
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def fake_cli(self, command, check):
        self.commands.append(command)
        if "download" in command:
            with zipfile.ZipFile(command[-1], "w") as archive:
                archive.writestr("remote_status.json", '{"returncode": 0}')
                archive.writestr("outputs/qexp_summary.csv", "pipeline_id\nB\n")

    def run_remote(self, **options):
        return execute_remote(self.config, project=PROJECT, colab_executable="colab", **options)

    def test_bundle_contains_only_code_and_portable_config(self):
        bundle = build_bundle(PROJECT, load_settings(self.config), self.root / "source.zip")
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            self.assertTrue(all(n.startswith("src/") or n in {"pyproject.toml", "configs/remote.json"} for n in names))
            config = json.loads(archive.read("configs/remote.json"))
            self.assertEqual(config["qexp"]["model_device"], "cuda")
            self.assertEqual(config["qexp"]["max_queries_per_dataset"], 3)
            self.assertEqual(config["output_dir"], "outputs")
        compile(launcher_source("pln-test", "run", True), "remote-launcher", "exec")

    def test_success_downloads_before_stopping_owned_session(self):
        with patch("qexp.colab.subprocess.run", side_effect=self.fake_cli):
            output = self.run_remote()
        operations = [c[3] for c in self.commands]
        self.assertEqual(operations, ["new", "upload", "exec", "download", "stop"])
        self.assertTrue((output / "outputs/qexp_summary.csv").exists())
        self.assertTrue(all(c[1:3] == ["--auth", "oauth2"] for c in self.commands))

    def test_download_failure_keeps_session(self):
        def fail_download(command, check):
            if "download" in command:
                self.commands.append(command)
                raise subprocess.CalledProcessError(1, command)
            self.fake_cli(command, check)
        with patch("qexp.colab.subprocess.run", side_effect=fail_download):
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_remote()
        self.assertNotIn("stop", [c[3] for c in self.commands])

    def test_kernel_error_is_detected_even_when_cli_succeeds(self):
        def kernel_error(command, check):
            self.fake_cli(command, check)
            if "download" in command:
                with zipfile.ZipFile(command[-1], "w") as archive:
                    archive.writestr("remote_status.json", '{"returncode": 1}')
        with patch("qexp.colab.subprocess.run", side_effect=kernel_error):
            with self.assertRaisesRegex(RuntimeError, "remota falhou"):
                self.run_remote()
        self.assertEqual(self.commands[-1][3], "stop")

    def test_existing_session_is_never_stopped(self):
        with patch("qexp.colab.subprocess.run", side_effect=self.fake_cli):
            self.run_remote(session="my-existing-session")
        self.assertEqual(self.commands[0][3], "status")
        self.assertNotIn("stop", [c[3] for c in self.commands])

    def test_dry_run_never_calls_cli_or_creates_output(self):
        with patch("qexp.colab.subprocess.run", side_effect=AssertionError("No remote access")):
            output = self.run_remote(dry_run=True)
        self.assertFalse(output.exists())

    def test_worker_packages_errors_and_completed_checkpoints(self):
        state = self.root / "state/checkpoints/toy"
        state.mkdir(parents=True)
        (state / "completed.json").write_text('{"query_id":"q1"}')
        with patch("qexp.remote_worker.subprocess.Popen", side_effect=OSError("pip failed")):
            returncode = run_worker(self.root, "run", self.root / "results.zip")
        self.assertEqual(returncode, 1)
        with zipfile.ZipFile(self.root / "results.zip") as archive:
            self.assertIn("state/checkpoints/toy/completed.json", archive.namelist())
            self.assertIn("pip failed", json.loads(archive.read("remote_status.json"))["error"])

    def test_restore_accepts_json_state_only_and_rejects_traversal(self):
        snapshot = self.root / "snapshot.zip"
        with zipfile.ZipFile(snapshot, "w") as archive:
            archive.writestr("state/generations/ok.json", '{}')
            archive.writestr("src/qexp/cli.py", 'raise RuntimeError()')
            archive.writestr("state/prepared/unsafe.pkl", b'pickle')
        target = self.root / "restored"
        restore_state(snapshot, target)
        self.assertTrue((target / "state/generations/ok.json").exists())
        self.assertFalse((target / "src").exists())
        self.assertFalse((target / "state/prepared").exists())
        with zipfile.ZipFile(snapshot, "w") as archive:
            archive.writestr("state/generations/../../../outside.json", '{}')
        with self.assertRaisesRegex(ValueError, "Caminho inválido"):
            restore_state(snapshot, target)


if __name__ == "__main__":
    unittest.main()

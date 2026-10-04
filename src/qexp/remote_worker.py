"""Worker enviado à VM Colab; nenhuma credencial é copiada para este processo."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import threading
import time
import zipfile


def restore_state(snapshot, root):
    """Restaura somente JSONs de estado, nunca scripts ou pickles externos."""
    with zipfile.ZipFile(snapshot) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if path.suffix != ".json" or path.parts[:2] not in {
                ("state", "generations"), ("state", "checkpoints")
            }:
                continue
            target = (root / path).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Caminho inválido no snapshot")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))


def collect_results(root, destination):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("remote_status.json", "experiment.log", "configs/remote.json"):
            path = root / name
            if path.exists():
                archive.write(path, name)
        for name in ("outputs", "state/generations", "state/checkpoints"):
            for path in sorted((root / name).rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(root).as_posix())


def run_worker(root, stage, destination, restore=None, resume=True):
    root = Path(root).resolve()
    status = {"returncode": 1, "stage": stage}
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "IR_DATASETS_HOME": str(root / "state/ir_datasets")}

    def execute(command, log, description):
        # Heartbeat também cobre pip e períodos sem saída do subprocesso.
        stopped = threading.Event()
        lock = threading.Lock()
        started = time.monotonic()

        def emit(text):
            with lock:
                print(text, end="", flush=True)
                log.write(text)
                log.flush()

        def heartbeat():
            while not stopped.wait(30):
                elapsed = time.monotonic() - started
                emit(f"[Colab {elapsed:.0f}s] {description}: comando em execução (não indica avanço).\n")

        emit(f"[Colab] {description}\n")
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            with subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    emit(line)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            stopped.set()
            thread.join()

    try:
        if restore is not None:
            restore_state(restore, root)
        with (root / "experiment.log").open("w", encoding="utf-8") as log:
            execute([sys.executable, "-m", "pip", "install", str(root)], log, "Instalando dependências")
            # Falha antes dos datasets/modelos quando a VM não disponibilizou CUDA.
            execute([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'Colab sem GPU CUDA'; print('GPU:', torch.cuda.get_device_name(0))"], log, "Verificando GPU")
            command = [sys.executable, "-u", "-m", "qexp.cli", stage, "--config", str(root / "configs/remote.json")]
            if not resume:
                command.append("--no-resume")
            execute(command, log, "Executando experimento")
        status["returncode"] = 0
    except Exception:
        status["error"] = traceback.format_exc()
        print(status["error"], flush=True)
    finally:
        (root / "remote_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        collect_results(root, destination)
    return status["returncode"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["run", "generate"])
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_worker(args.root, args.stage, args.archive, args.restore, not args.no_resume))


if __name__ == "__main__":
    main()

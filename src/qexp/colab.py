"""Executa a expansão em GPU remota usando a CLI oficial google-colab-cli."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

from .cli import load_settings
from .config import DATASETS


def remote_config(settings):
    """Preserva o protocolo, mas substitui caminhos locais por caminhos na VM."""
    return {
        "project_root": "..", "datasets": settings.get("datasets", [d["slug"] for d in DATASETS]),
        "output_dir": "outputs", "prepared_dir": "state/prepared",
        "checkpoint_dir": "state/checkpoints",
        "qexp": {**settings["qexp"], "model_device": "cuda", "cache_dir": "state/generations"},
    }


def build_bundle(project, settings, destination):
    project = Path(project).resolve()
    files = [project / "pyproject.toml", *sorted((project / "src").rglob("*.py"))]
    if not (project / "src/qexp/remote_worker.py").is_file():
        raise ValueError("project deve conter src/qexp/remote_worker.py")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(project):
                raise ValueError(f"Arquivo fora do projeto: {path}")
            archive.write(path, path.relative_to(project).as_posix())
        archive.writestr("configs/remote.json", json.dumps(remote_config(settings), indent=2))
    return Path(destination)


def launcher_source(job, stage, restore=False, resume=True):
    # repr serializa valores como literais Python; nenhum parâmetro passa por shell.
    return f'''from pathlib import Path
import subprocess
import sys
import zipfile

root = Path({('/content/' + job)!r})
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile({('/content/' + job + '-source.zip')!r}) as archive:
    archive.extractall(root)
command = [sys.executable, str(root / 'src/qexp/remote_worker.py'),
           '--root', str(root), '--stage', {stage!r},
           '--archive', {('/content/' + job + '-results.zip')!r}]
if {restore!r}:
    command += ['--restore', {('/content/' + job + '-restore.zip')!r}]
if not {resume!r}:
    command.append('--no-resume')
subprocess.run(command, check=True)
'''


def find_colab():
    sibling = Path(sys.executable).parent / "colab"
    executable = str(sibling) if sibling.is_file() else shutil.which("colab")
    if not executable:
        raise RuntimeError('Instale a CLI oficial: python -m pip install google-colab-cli')
    return executable


def execute_remote(config_path, *, project=None, gpu="A100", stage="run", session=None,
                   keep=False, restore=None, resume=True, timeout=43200, dry_run=False,
                   high_mem=False, colab_executable=None):
    if stage not in {"run", "generate"}:
        raise ValueError("Use run ou generate remotamente; analyze pode executar localmente.")
    if gpu not in {"T4", "L4", "A100", "H100", "G4"} or timeout <= 0:
        raise ValueError("GPU ou timeout inválido")
    settings = load_settings(config_path)
    # Valida antes de alocar recursos pagos.
    from .selection import qexp_validate_config
    qexp_validate_config(settings["qexp"])
    if not settings.get("datasets") or set(settings["datasets"]) - {d["slug"] for d in DATASETS}:
        raise ValueError("Datasets inválidos")
    if restore is not None:
        restore = Path(restore).resolve()
        with zipfile.ZipFile(restore) as archive:
            if archive.testzip() is not None:
                raise ValueError("Snapshot de retomada corrompido")
    project = Path(project or Path(__file__).resolve().parents[2])
    job = "pln-" + uuid.uuid4().hex[:12]
    session_name = session or job
    output = Path(settings["output_dir"]) / "colab" / job
    executable = colab_executable or ("colab" if dry_run else find_colab())
    prefix = [executable, "--auth", "oauth2"]
    created = False
    downloaded = False

    def call(*args):
        command = [*prefix, *map(str, args)]
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True)

    with tempfile.TemporaryDirectory(prefix="pln-colab-") as temporary:
        temporary = Path(temporary)
        bundle = build_bundle(project, settings, temporary / "source.zip")
        launcher = temporary / "launch.py"
        launcher.write_text(launcher_source(job, stage, restore is not None, resume), encoding="utf-8")
        print(f"GPU solicitada: {gpu}. Sessão: {session_name}. Destino local: {output}")
        print("A execução remota usa as unidades de computação da conta autenticada.")
        try:
            if session is None:
                call("new", "-s", session_name, "--gpu", gpu, *(["--high-mem"] if high_mem else []))
                created = True
            else:
                call("status", "-s", session_name)
            call("upload", "-s", session_name, bundle, f"content/{job}-source.zip")
            if restore is not None:
                call("upload", "-s", session_name, restore, f"content/{job}-restore.zip")
            # Exec pode retornar zero mesmo se o kernel reportar um erro. O status no ZIP é obrigatório.
            execution_error = None
            try:
                call("exec", "-s", session_name, "--timeout", timeout, "-f", launcher)
            except subprocess.CalledProcessError as error:
                execution_error = error
            if not dry_run:
                output.mkdir(parents=True, exist_ok=False)
            snapshot = output / "remote-results.zip"
            call("download", "-s", session_name, f"content/{job}-results.zip", snapshot)
            if not dry_run:
                with zipfile.ZipFile(snapshot) as archive:
                    if archive.testzip() is not None:
                        raise RuntimeError("Download corrompido; a sessão será preservada.")
                    status = json.loads(archive.read("remote_status.json"))
                    # Extração restrita à pasta nova desta execução.
                    for name in archive.namelist():
                        target = (output / name).resolve()
                        if not target.is_relative_to(output.resolve()):
                            raise ValueError("Caminho inválido no ZIP de resultados")
                    archive.extractall(output)
                downloaded = True
                if status.get("returncode") != 0:
                    raise RuntimeError(f"Execução remota falhou. Consulte {output / 'experiment.log'} e remote_status.json")
                if execution_error:
                    raise execution_error
            else:
                downloaded = True
        finally:
            if created and downloaded and not keep:
                call("stop", "-s", session_name)
            elif created or session is not None:
                print(f"Sessão preservada: {session_name}. Para encerrar: "
                      + shlex.join([*prefix, "stop", "-s", session_name]))
                print(f"Arquivo remoto de recuperação, quando concluído: content/{job}-results.zip")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "generate"], nargs="?", default="run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", choices=["T4", "L4", "A100", "H100", "G4"], default="A100", help="GPU solicitada ao criar uma sessão (padrão: A100)")
    parser.add_argument("--session", help="Reutiliza uma sessão existente; não a encerra nem troca sua GPU")
    parser.add_argument("--keep", action="store_true", help="Mantém a sessão após o download")
    parser.add_argument("--restore", type=Path, help="remote-results.zip anterior para restaurar checkpoints/gerações")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--high-mem", action="store_true")
    parser.add_argument("--timeout", type=float, default=43200, help="Timeout do comando remoto em segundos")
    parser.add_argument("--dry-run", action="store_true", help="Exibe comandos sem autenticar, enviar arquivos ou alocar GPU")
    args = parser.parse_args()
    output = execute_remote(args.config, gpu=args.gpu, stage=args.stage, session=args.session,
                            keep=args.keep, restore=args.restore, resume=not args.no_resume,
                            high_mem=args.high_mem, timeout=args.timeout, dry_run=args.dry_run)
    print("Destino dos resultados:", output)


if __name__ == "__main__":
    main()

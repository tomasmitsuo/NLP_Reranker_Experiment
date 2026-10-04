"""Progresso Rich em terminais/notebooks e mensagens legíveis no stream do Colab."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

_current = ContextVar("qexp_progress", default=None)


class ProgressReporter:
    def __init__(self, output_dir, mode="auto", *, console=None, heartbeat_sec=30):
        if mode not in {"auto", "plain", "off"}:
            raise ValueError("Progresso deve ser auto, plain ou off")
        self.mode = mode
        self.console = console or Console(stderr=True)
        self.interactive = mode == "auto" and ((self.console.is_terminal and not self.console.is_dumb_terminal) or self.console.is_jupyter)
        self.path = Path(output_dir) / "qexp_progress.json"
        self.heartbeat_sec = heartbeat_sec
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.state = dict(status="running", phase="Inicializando", dataset="", query_id="",
                          total_queries=None, completed_queries=0, executed_queries=0, reused_queries=0)
        self.display = None
        self.thread = None

    def __enter__(self):
        self.token = _current.set(self)  # Também suprime tqdm legado quando mode=off.
        if self.mode == "off":
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.interactive:
            self.display = Progress(
                TextColumn("{task.description}", markup=False), BarColumn(), TaskProgressColumn(),
                TextColumn("{task.fields[counts]}"), TextColumn("{task.fields[reuse]}"), TimeElapsedColumn(), TimeRemainingColumn(),
                console=self.console, auto_refresh=False,
            )
            self.task = self.display.add_task("Preparação", total=None, counts="", reuse="")
            self.display.start()
        self.update()
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="qexp-progress")
        self.thread.start()
        return self

    def _heartbeat_loop(self):
        while not self.stop_event.wait(self.heartbeat_sec):
            self.update(heartbeat=True)

    def update(self, phase=None, *, complete=False, reused=False, heartbeat=False, **fields):
        if self.mode == "off":
            return
        with self.lock:
            if phase is not None:
                self.state["phase"] = phase
            self.state.update(fields)
            if complete:
                self.state["completed_queries"] += 1
                self.state["reused_queries" if reused else "executed_queries"] += 1
            elapsed = time.monotonic() - self.started
            self.state.update(elapsed_sec=round(elapsed, 1), updated_at=datetime.now(timezone.utc).isoformat())
            total = self.state["total_queries"]
            done = self.state["completed_queries"]
            self.state["percent"] = round(100 * done / total, 1) if total else None
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            description = " | ".join(str(self.state[key]) for key in ("dataset", "query_id", "phase") if self.state[key])
            counts = f"queries {done}/{total} ({self.state['percent']}%)" if total else "preparação/análise"
            counts += f" | novas={self.state['executed_queries']} cache={self.state['reused_queries']}"
            if self.display:
                self.display.update(self.task, description=description, total=total, completed=done, counts=f"{done}/{total}" if total else "total pendente", reuse="cache=" + str(self.state["reused_queries"]), refresh=True)
            else:
                alive = " | processo ativo (sem avanço confirmado)" if heartbeat else ""
                self.console.print(f"[progresso {elapsed:.0f}s] {counts} | {description}{alive}",
                                   markup=False, highlight=False, soft_wrap=True)
                self.console.file.flush()

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        try:
            if self.mode != "off":
                if exc is None:
                    self.update("Concluído", status="completed")
                else:
                    self.update("Falhou: " + self.state["phase"], status="failed", error=f"{exc_type.__name__}: {exc}")
                if self.display:
                    self.display.stop()
        finally:
            _current.reset(self.token)


def active():
    return _current.get() is not None


def report(phase=None, **fields):
    reporter = _current.get()
    if reporter is not None:
        reporter.update(phase, **fields)


@contextmanager
def activity(description):
    """Mostra carregamentos lentos e restaura a etapa anterior ao terminar."""
    reporter = _current.get()
    previous = reporter.state["phase"] if reporter else None
    report(description)
    try:
        yield
    except BaseException:
        raise
    else:
        if previous is not None:
            report(previous)

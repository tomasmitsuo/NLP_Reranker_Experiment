"""Analisa uma sessão já baixada e salva seus gráficos em <sessão>/images."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require_plotting():
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError('Instale os gráficos antes de executar: python -m pip install -e ".[colab]"') from error


def analyze_session(session):
    from .checkpoints import load_result
    from .selector_features import query_hash
    from .selector_study import pipeline_diagnostics

    require_plotting()
    session = Path(session).resolve()
    snapshot = session / "outputs/qexp_run.json"
    if not snapshot.is_file():
        raise FileNotFoundError(f"Snapshot da sessão não encontrado: {snapshot}")
    status_path = session / "remote_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("returncode") != 0:
        raise ValueError("A sessão remota falhou; não gerar relatório de execução concluída.")
    result = load_result(snapshot)
    config = result["metadata"]["config"]
    # Todas as consultas efetivamente executadas; nenhuma nova partição/treino.
    frame = result["metrics"].query("pipeline_id == 'B'")[["dataset", "query_id", "query_text"]].copy()
    if frame.empty:
        raise ValueError("Snapshot sem baseline B.")
    if result["metrics"].duplicated(["dataset", "query_id", "pipeline_id"]).any():
        raise ValueError("Métricas duplicadas na sessão.")
    expected = frame.set_index(["dataset", "query_id"]).index
    for method, rows in result["metrics"].groupby("pipeline_id"):
        if set(rows.set_index(["dataset", "query_id"]).index) != set(expected):
            raise ValueError(f"Consultas não pareadas: {method}")
    frame["group"] = frame.query_text.map(query_hash)
    frame["split"] = "test"  # Seleção de linhas da função; não cria um teste científico novo.
    print(f"Analisando {len(frame)} consultas da sessão; imagens em {session / 'images'}", flush=True)
    comparisons, queries = pipeline_diagnostics(result, frame, seed=config["seed"])
    from .naming import rename_table
    public = config.get("id_scheme") == "technique_selector"
    analysis = session / "analysis"
    analysis.mkdir(exist_ok=True)
    (rename_table(comparisons) if public else comparisons).to_csv(analysis / "pipeline_comparisons.csv", index=False)
    (rename_table(queries) if public else queries).to_csv(analysis / "pipeline_queries.csv", index=False)
    (rename_table(result["summary"]) if public else result["summary"]).to_csv(analysis / "pipeline_summary.csv", index=False)
    manifest = {"status": "analyzing", "source": "../outputs/qexp_run.json",
                "source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "queries": len(frame), "pipelines": sorted(result["metrics"].pipeline_id.unique()),
                "scope": "Todas as consultas executadas nesta sessão; sem retreinar seletores.",
                "metric": f"ndcg@{config['eval_k']}", "seed": config["seed"],
                "statistics": "Bootstrap por consulta normalizada: 2000 reamostragens; permutação pareada: teste exato para até 15 grupos não nulos, ou 9999 sorteios; Holm em todos os contrastes não-B desta tabela."}
    manifest_path = analysis / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    try:
        render_session(result, comparisons, queries, session / "images")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        raise
    lines = ["# Análise da sessão de GPU", "", manifest["scope"], "",
             "[Galeria de imagens](../images/gallery.html)", "",
             "## Qualidade e custo", "", result["summary"].to_markdown(index=False), "",
             "## Diferenças pareadas frente a B", "", comparisons.to_markdown(index=False), "",
             manifest["statistics"], "",
             "ICs são marginais de 95%; ausência de significância não demonstra equivalência.",
             "Os custos são compostos e podem incluir tempos históricos de cache; não são latências independentes.",
             "G/K preservam candidatos de B; C/D/F podem modificar a recuperação. A não usa BGE.",
             "O treino/validação/teste do seletor é um estudo separado; este relatório não cria novas partições."]
    if public:
        from .naming import public_text, public_id
        lines = [public_text(line) for line in lines]
        manifest["pipelines"] = list(map(public_id, manifest["pipelines"]))
    (analysis / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Análise concluída: {analysis / 'report.md'}", flush=True)
    return session / "images"


def render_session(result, comparisons, queries, images):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images.mkdir(exist_ok=True)
    metric = f"ndcg@{result['metadata']['config']['eval_k']}"
    datasets = sorted(comparisons.dataset.unique())
    saved = []
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 180})

    def panels(height=4.5):
        return plt.subplots(1, len(datasets), figsize=(5 * len(datasets), height), squeeze=False)

    def save(fig, name, caption):
        if result["metadata"]["config"].get("id_scheme") == "technique_selector":
            import re
            from .naming import public_id
            def renamed(value):
                return re.sub(r'\b(C_H4|E_RA|[ABCDEFGKLR])\b', lambda m: public_id(m[0]), value)
            for axis in fig.axes:
                xlabels, ylabels = [t.get_text() for t in axis.get_xticklabels()], [t.get_text() for t in axis.get_yticklabels()]
                if any(public_id(t) != t for t in xlabels):
                    axis.set_xticks(axis.get_xticks(), [public_id(t) for t in xlabels])
                if any(public_id(t) != t for t in ylabels):
                    axis.set_yticks(axis.get_yticks(), [public_id(t) for t in ylabels])
                axis.set_xlabel(renamed(axis.get_xlabel()))
                axis.set_ylabel(renamed(axis.get_ylabel()))
                legend = axis.get_legend()
                if legend:
                    for t in legend.get_texts():
                        t.set_text(renamed(t.get_text()))
                for t in axis.texts:
                    t.set_text(renamed(t.get_text()))
            for legend in fig.legends:
                for t in legend.get_texts():
                    t.set_text(renamed(t.get_text()))
            caption = renamed(caption)
        fig.tight_layout(rect=(0, .05, 1, .9 if fig.legends else .98))
        fig.text(.01, .01, f"{metric} · Todas as consultas executadas na sessão", fontsize=8)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(images / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
        saved.append((name, caption))

    fig, axes = panels()
    for ax, dataset in zip(axes[0], datasets):
        rows = comparisons[comparisons.dataset == dataset].sort_values("pipeline_id")
        ax.bar(rows.pipeline_id, rows.ndcg, color=["#222222" if m == "B" else "#0072B2" for m in rows.pipeline_id])
        ax.set(title=dataset, ylabel=metric, ylim=(0, 1))
        ax.tick_params(axis="x", rotation=45)
    save(fig, "quality", "Qualidade média de cada pipeline nas mesmas consultas da sessão. B: BM25 + BGE.")

    fig, axes = panels()
    limits = (min(0, comparisons.ci95_low.min()), max(0, comparisons.ci95_high.max()))
    margin = max((limits[1] - limits[0]) * .08, .005)
    for ax, dataset in zip(axes[0], datasets):
        rows = comparisons[(comparisons.dataset == dataset) & (comparisons.pipeline_id != "B")].sort_values("pipeline_id")
        for i, row in enumerate(rows.itertuples()):
            color = "#009E73" if row.pipeline_id in {"G", "K"} else "#0072B2"
            ax.plot([row.ci95_low, row.ci95_high], [i, i], color=color)
            ax.scatter(row.mean_delta, i, color=color, marker="s" if row.p_holm < .05 else "o")
        ax.axvline(0, color="black", lw=.8, ls="--")
        ax.set(title=dataset, yticks=np.arange(len(rows)), yticklabels=rows.pipeline_id,
               xlabel="Δ nDCG frente a B (IC95%)", xlim=(limits[0] - margin, limits[1] + margin))
    save(fig, "effects", "IC95% marginal por consulta normalizada. Quadrado: p Holm < 0,05. Verde: G/K, candidatos fixos.")

    fig, axes = panels()
    for ax, dataset in zip(axes[0], datasets):
        rows = comparisons[(comparisons.dataset == dataset) & (comparisons.pipeline_id != "B")].sort_values("pipeline_id")
        ax.barh(rows.pipeline_id, rows.harm_rate, color="#D55E00", label="Piora")
        ax.barh(rows.pipeline_id, rows.tie_rate, left=rows.harm_rate, color="#cccccc", label="Empate")
        ax.barh(rows.pipeline_id, rows.gain_rate, left=rows.harm_rate + rows.tie_rate, color="#0072B2", label="Melhora")
        ax.set(title=dataset, xlim=(0, 1), xlabel="Proporção das consultas")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3)
    save(fig, "wins_ties_losses", "Frequência de melhorias, empates e pioras em relação a B.")

    fig, axes = panels()
    for ax, dataset in zip(axes[0], datasets):
        rows = comparisons[comparisons.dataset == dataset]
        for row in rows.itertuples():
            ax.scatter(row.estimated_sec, row.ndcg, marker="X" if row.pipeline_id == "B" else "o")
            ax.annotate(row.pipeline_id, (row.estimated_sec, row.ndcg), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(title=dataset, xlabel="Custo composto (s/consulta)", ylabel=metric)
        ax.grid(alpha=.2)
    save(fig, "quality_cost", "Qualidade versus custo composto histórico. Escalas por coleção; não é medição de latência independente.")

    fig, axes = panels()
    for ax, dataset in zip(axes[0], datasets):
        for method, rows in queries[(queries.dataset == dataset) & (queries.pipeline_id != "B")].groupby("pipeline_id"):
            values = np.sort(rows.delta.to_numpy())
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=method)
        ax.axvline(0, color="black", lw=.8, ls="--")
        ax.set(title=dataset, xlabel="Δ nDCG frente a B", ylabel="Fração acumulada de consultas", ylim=(0, 1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(9, len(labels)))
    save(fig, "delta_distribution", "Distribuição acumulada dos deltas: à esquerda de zero estão as consultas degradadas.")

    fig, axes = panels()
    for ax, dataset in zip(axes[0], datasets):
        rows = result["summary"][result["summary"].dataset == dataset].sort_values("pipeline_id")
        x = np.arange(len(rows))
        ax.bar(x - .2, rows.expansion_rate, width=.4, label="Solicitada", color="#0072B2")
        ax.bar(x + .2, rows.valid_expansion_rate, width=.4, label="Aplicada", color="#009E73")
        ax.set(title=dataset, xticks=x, xticklabels=rows.pipeline_id, ylim=(0, 1), ylabel="Proporção das consultas")
        ax.tick_params(axis="x", rotation=45)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2)
    save(fig, "expansion_rates", "Solicitação e aplicação efetiva de expansão. Aplicação não significa benefício nem validade semântica.")

    body = ["<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>Resultados da sessão GPU</title>",
            "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px}img{max-width:100%}figure{margin:40px 0}</style>",
            "<h1>Resultados da sessão GPU</h1><p><a href='../analysis/report.md'>Relatório estatístico</a> · <a href='../outputs/qexp_report.md'>Relatório do experimento</a></p>"]
    for name, caption in saved:
        body.append(f"<figure><img src='{name}.png' alt='{html.escape(caption, quote=True)}'><figcaption>{html.escape(caption)} "
                    f"<a href='{name}.pdf'>PDF</a> · <a href='{name}.svg'>SVG</a> · <a href='{name}.png'>PNG</a></figcaption></figure>")
    (images / "gallery.html").write_text("\n".join(body) + "</html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()
    print(analyze_session(args.session))


if __name__ == "__main__":
    main()

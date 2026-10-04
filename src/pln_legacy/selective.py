import time

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TOKEN_RE = re.compile(r"(?u)\b\w+\b")
TARGET_THRESHOLD = 1e-12


def classifier_tokens(text):
    return TOKEN_RE.findall(str(text).lower())


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def softmax_entropy(values):
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum()
    return float(entropy / np.log(len(values))) if len(values) > 1 else 0.0


def query_features(query):
    tokens = classifier_tokens(query)
    char_count = len(query)
    return {
        "query_char_count": char_count,
        "query_word_count": len(tokens),
        "query_unique_ratio": safe_ratio(len(set(tokens)), len(tokens)),
        "query_avg_word_length": safe_ratio(sum(map(len, tokens)), len(tokens)),
        "query_digit_ratio": safe_ratio(sum(c.isdigit() for c in query), char_count),
        "query_punctuation_ratio": safe_ratio(
            sum(not c.isalnum() and not c.isspace() for c in query), char_count
        ),
    }


def ranking_features(group):
    ordered = group.sort_values("rank")
    scores = ordered["score"].astype(float).to_numpy()
    top10_scores = scores[:10]
    query_token_set = set(classifier_tokens(ordered["query_text"].iloc[0]))
    document_token_sets = [
        set(classifier_tokens(text)) for text in ordered["doc_text"].fillna("")
    ]
    coverages = np.array([
        safe_ratio(len(query_token_set & tokens), len(query_token_set))
        for tokens in document_token_sets
    ])
    document_lengths = np.array([len(tokens) for tokens in document_token_sets], dtype=float)
    return {
        "bm25_score_max": scores.max(),
        "bm25_score_mean": scores.mean(),
        "bm25_score_std": scores.std(),
        "bm25_score_range": scores.max() - scores.min(),
        "bm25_top1_top2_margin": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        "bm25_top1_top5_margin": scores[0] - scores[min(4, len(scores) - 1)],
        "bm25_top10_std": top10_scores.std(),
        "bm25_top10_entropy": softmax_entropy(top10_scores),
        "bm25_top1_mean_ratio": safe_ratio(scores[0], scores.mean()),
        "bm25_score_slope": np.polyfit(np.arange(1, len(scores) + 1), scores, 1)[0]
        if len(scores) > 1 else 0.0,
        "lexical_coverage_top1": coverages[0],
        "lexical_coverage_mean": coverages.mean(),
        "lexical_coverage_max": coverages.max(),
        "lexical_coverage_std": coverages.std(),
        "lexical_coverage_top3_mean": coverages[:3].mean(),
        "doc_word_count_mean": document_lengths.mean(),
        "doc_word_count_std": document_lengths.std(),
    }


def build_pre_reranking_features(rankings):
    required = {"dataset", "pipeline", "query_id", "query_text", "rank", "score", "doc_text"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no ranking: {sorted(missing)}")
    rows = []
    bm25 = rankings[rankings["pipeline"] == "BM25"]
    for (dataset, query_id), group in bm25.groupby(["dataset", "query_id"], sort=False):
        query = str(group["query_text"].iloc[0])
        rows.append({
            "dataset": dataset,
            "query_id": str(query_id),
            "query_text": query,
            **query_features(query),
            **ranking_features(group),
        })
    return pd.DataFrame(rows)


def build_classifier_dataset(rankings, metric_deltas):
    features = build_pre_reranking_features(rankings)
    labels = metric_deltas[["dataset", "query_id", "delta_nDCG@10"]].copy()
    labels["query_id"] = labels["query_id"].astype(str)
    data = features.merge(labels, on=["dataset", "query_id"], validate="one_to_one")
    data["rerank_beneficial"] = (data["delta_nDCG@10"] > TARGET_THRESHOLD).astype(int)
    return data


def make_query_classifier(numeric_columns):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    features = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("query_tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2,
            max_features=3000, sublinear_tf=True,
        ), "query_text"),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42,
        )),
    ])


def classifier_evaluation_summary(frame, probabilities, evaluation):
    target = frame["rerank_beneficial"].to_numpy()
    predicted = (probabilities >= 0.5).astype(int)
    deltas = frame["delta_nDCG@10"].to_numpy()
    selected_delta = deltas * predicted
    positive_gain = np.clip(deltas, 0, None)
    return {
        "evaluation": evaluation,
        "queries": len(frame),
        "positive_rate": target.mean(),
        "roc_auc": roc_auc_score(target, probabilities),
        "average_precision": average_precision_score(target, probabilities),
        "balanced_accuracy": balanced_accuracy_score(target, predicted),
        "precision": precision_score(target, predicted, zero_division=0),
        "recall": recall_score(target, predicted, zero_division=0),
        "f1": f1_score(target, predicted, zero_division=0),
        "rerank_rate": predicted.mean(),
        "mean_delta_always_rerank": deltas.mean(),
        "mean_delta_classifier": selected_delta.mean(),
        "mean_delta_oracle": positive_gain.mean(),
        "gain_retention_vs_always": safe_ratio(selected_delta.mean(), deltas.mean()),
        "reranker_calls_saved_ratio": 1.0 - predicted.mean(),
        "captured_positive_gain_ratio": safe_ratio(
            (positive_gain * predicted).sum(), positive_gain.sum()
        ),
        "harmful_queries_avoided_ratio": safe_ratio(
            ((deltas < -TARGET_THRESHOLD) & (predicted == 0)).sum(),
            (deltas < -TARGET_THRESHOLD).sum(),
        ),
    }


def evaluate_query_classifier(frame):
    excluded = {"dataset", "query_id", "query_text", "delta_nDCG@10", "rerank_beneficial"}
    numeric_columns = [column for column in frame.columns if column not in excluded]
    x = frame[["query_text", *numeric_columns]]
    y = frame["rerank_beneficial"]
    evaluations = [
        ("stratified_5_fold", StratifiedKFold(5, shuffle=True, random_state=42), None),
        ("leave_one_dataset_out", GroupKFold(frame["dataset"].nunique()), frame["dataset"]),
    ]
    summaries = []
    prediction_frames = []
    for name, cv, groups in evaluations:
        probabilities = cross_val_predict(
            make_query_classifier(numeric_columns), x, y, groups=groups,
            cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1]
        predictions = frame[[
            "dataset", "query_id", "query_text", "delta_nDCG@10", "rerank_beneficial"
        ]].copy()
        predictions["evaluation"] = name
        predictions["rerank_probability"] = probabilities
        predictions["rerank_predicted"] = (probabilities >= 0.5).astype(int)
        prediction_frames.append(predictions)
        summaries.append(classifier_evaluation_summary(frame, probabilities, name))
    return pd.DataFrame(summaries), pd.concat(prediction_frames, ignore_index=True)


def predict_reranking_need(rankings, model, threshold=0.5):
    features = build_pre_reranking_features(rankings)
    probabilities = model.predict_proba(features[list(model.feature_names_in_)])[:, 1]
    result = features[["dataset", "query_id", "query_text"]].copy()
    result["rerank_probability"] = probabilities
    result["rerank_predicted"] = (probabilities >= threshold).astype(int)
    return result


def run_query_classifier_experiment(rankings, metric_deltas, output_dir="."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_classifier_dataset(rankings, metric_deltas)
    summary, predictions = evaluate_query_classifier(data)
    excluded = {"dataset", "query_id", "query_text", "delta_nDCG@10", "rerank_beneficial"}
    numeric_columns = [column for column in data.columns if column not in excluded]
    model = make_query_classifier(numeric_columns)
    model.fit(data[["query_text", *numeric_columns]], data["rerank_beneficial"])
    feature_names = model.named_steps["features"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    coefficient_table = pd.DataFrame({
        "feature": feature_names, "coefficient": coefficients,
    }).sort_values("coefficient", key=lambda values: values.abs(), ascending=False)
    data.to_csv(output_dir / "query_classifier_dataset.csv", index=False)
    summary.to_csv(output_dir / "query_classifier_evaluation.csv", index=False)
    predictions.to_csv(output_dir / "query_classifier_predictions.csv", index=False)
    coefficient_table.to_csv(output_dir / "query_classifier_coefficients.csv", index=False)
    joblib.dump(model, output_dir / "query_reranking_classifier.joblib")
    metadata = {
        "target": "delta_nDCG@10 > 0",
        "decision_threshold": 0.5,
        "numeric_features": numeric_columns,
        "important_constraint": "Todas as features existem antes do rerankeamento.",
    }
    (output_dir / "query_classifier_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"dataset": data, "summary": summary, "predictions": predictions, "model": model}

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

QUERY_ONLY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUERY_ONLY_SEED = 42


def build_query_only_data(rankings, metric_deltas):
    queries = rankings[["dataset", "query_id", "query_text"]].drop_duplicates(
        ["dataset", "query_id"]
    ).copy()
    queries["query_id"] = queries["query_id"].astype(str)
    labels = metric_deltas[["dataset", "query_id", "delta_nDCG@10"]].copy()
    labels["query_id"] = labels["query_id"].astype(str)
    data = queries.merge(labels, on=["dataset", "query_id"], validate="one_to_one")
    data["rerank_beneficial"] = (data["delta_nDCG@10"] > TARGET_THRESHOLD).astype(int)
    linguistic = pd.DataFrame(
        [query_features(str(query)) for query in data["query_text"]],
        index=data.index,
    )
    return pd.concat([data, linguistic], axis=1)


def query_only_model_specs():
    balanced_logistic = lambda: LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=QUERY_ONLY_SEED
    )
    return {
        "word_tfidf_logistic": ("text", Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True
            )),
            ("classifier", balanced_logistic()),
        ]), False),
        "char_tfidf_logistic": ("text", Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                max_features=8000, sublinear_tf=True,
            )),
            ("classifier", balanced_logistic()),
        ]), False),
        "linguistic_gradient_boosting": ("linguistic", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
                l2_regularization=1.0, class_weight="balanced",
                random_state=QUERY_ONLY_SEED,
            )),
        ]), False),
        "embedding_logistic": ("embedding", Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", C=0.1, max_iter=2000,
                random_state=QUERY_ONLY_SEED,
            )),
        ]), False),
        "embedding_mlp": ("embedding", Pipeline([
            ("scale", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64,), alpha=1.0, max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=25, random_state=QUERY_ONLY_SEED,
            )),
        ]), True),
    }


def balance_training_fold(x, y, seed):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    target_size = max(map(len, groups))
    indices = np.concatenate([
        np.concatenate([group, rng.choice(group, target_size - len(group), replace=True)])
        if len(group) < target_size else group
        for group in groups
    ])
    rng.shuffle(indices)
    return (x.iloc[indices] if hasattr(x, "iloc") else x[indices]), y[indices]


def benchmark_input(data, input_type, linguistic_columns, embeddings):
    if input_type == "text":
        return data["query_text"]
    if input_type == "linguistic":
        return data[linguistic_columns]
    return embeddings


def evaluate_query_only_benchmark(data, embeddings):
    y = data["rerank_beneficial"].to_numpy()
    linguistic_columns = list(query_features("").keys())
    split_sets = {
        "stratified_5_fold": list(StratifiedKFold(
            5, shuffle=True, random_state=QUERY_ONLY_SEED
        ).split(data, y)),
        "leave_one_dataset_out": list(GroupKFold(
            data["dataset"].nunique()
        ).split(data, y, groups=data["dataset"])),
    }
    summary_rows, prediction_frames, training_history_rows = [], [], []
    for model_name, (input_type, template, oversample) in query_only_model_specs().items():
        x = benchmark_input(data, input_type, linguistic_columns, embeddings)
        for evaluation, splits in split_sets.items():
            probabilities = np.zeros(len(data))
            for fold, (train_idx, test_idx) in enumerate(splits):
                x_train = x.iloc[train_idx] if hasattr(x, "iloc") else x[train_idx]
                x_test = x.iloc[test_idx] if hasattr(x, "iloc") else x[test_idx]
                y_train = y[train_idx]
                if oversample:
                    x_train, y_train = balance_training_fold(
                        x_train, y_train, QUERY_ONLY_SEED + fold
                    )
                model = clone(template).fit(x_train, y_train)
                if model_name == "embedding_mlp":
                    classifier = model.named_steps["classifier"]
                    loss_curve = getattr(classifier, "loss_curve_", [])
                    validation_scores = getattr(classifier, "validation_scores_", [])
                    for epoch in range(max(len(loss_curve), len(validation_scores))):
                        training_history_rows.append({
                            "model": model_name,
                            "evaluation": evaluation,
                            "fold": fold,
                            "epoch": epoch + 1,
                            "loss": loss_curve[epoch] if epoch < len(loss_curve) else np.nan,
                            "validation_score": (
                                validation_scores[epoch]
                                if epoch < len(validation_scores) else np.nan
                            ),
                            "n_iter": getattr(classifier, "n_iter_", np.nan),
                            "train_size": len(y_train),
                            "test_size": len(test_idx),
                        })
                probabilities[test_idx] = model.predict_proba(x_test)[:, 1]
            metrics = classifier_evaluation_summary(data, probabilities, evaluation)
            metrics.update({"model": model_name, "input": input_type})
            summary_rows.append(metrics)
            fold_predictions = data[[
                "dataset", "query_id", "query_text",
                "delta_nDCG@10", "rerank_beneficial",
            ]].copy()
            fold_predictions["model"] = model_name
            fold_predictions["evaluation"] = evaluation
            fold_predictions["rerank_probability"] = probabilities
            fold_predictions["rerank_predicted"] = (probabilities >= 0.5).astype(int)
            prediction_frames.append(fold_predictions)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(training_history_rows),
    )


def run_query_only_benchmark(rankings, metric_deltas, output_dir="."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_query_only_data(rankings, metric_deltas)
    print(f"Gerando embeddings para {len(data)} queries...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(QUERY_ONLY_EMBEDDING_MODEL)
    embedding_start = time.perf_counter()
    embeddings = encoder.encode(
        data["query_text"].tolist(), batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    embedding_elapsed_sec = time.perf_counter() - embedding_start
    timing = {
        "embedding_encode_total_sec": embedding_elapsed_sec,
        "embedding_encode_avg_sec": safe_ratio(embedding_elapsed_sec, len(data)),
    }
    summary, predictions, training_history = evaluate_query_only_benchmark(data, embeddings)
    data.to_csv(output_dir / "query_only_classifier_dataset.csv", index=False)
    summary.to_csv(output_dir / "query_only_model_benchmark.csv", index=False)
    predictions.to_csv(output_dir / "query_only_model_predictions.csv", index=False)
    training_history.to_csv(output_dir / "query_only_mlp_training_history.csv", index=False)
    np.save(output_dir / "query_only_embeddings.npy", embeddings)
    return {
        "dataset": data,
        "summary": summary,
        "predictions": predictions,
        "training_history": training_history,
        "embeddings": embeddings,
        "timing": timing,
    }

def measure_query_only_inference_time(data, embeddings, embedding_encode_avg_sec=0.0):
    y = data["rerank_beneficial"].to_numpy()
    linguistic_columns = list(query_features("").keys())
    rows = []

    for model_name, (input_type, template, oversample) in query_only_model_specs().items():
        x = benchmark_input(data, input_type, linguistic_columns, embeddings)
        x_train, y_train = x, y
        if oversample:
            x_train, y_train = balance_training_fold(x_train, y_train, QUERY_ONLY_SEED)

        model = clone(template).fit(x_train, y_train)
        start = time.perf_counter()
        _ = model.predict_proba(x)[:, 1]
        predict_elapsed_sec = time.perf_counter() - start
        predict_avg_sec = safe_ratio(predict_elapsed_sec, len(data))
        embedding_avg_sec = embedding_encode_avg_sec if input_type == "embedding" else 0.0

        rows.append({
            "model": model_name,
            "input": input_type,
            "queries": len(data),
            "predict_total_sec": predict_elapsed_sec,
            "predict_avg_sec": predict_avg_sec,
            "embedding_encode_avg_sec": embedding_avg_sec,
            "classifier_inference_avg_sec": predict_avg_sec + embedding_avg_sec,
        })

    return pd.DataFrame(rows)


def build_query_only_policy_audit(query_only_benchmark, results_all):
    data = query_only_benchmark["dataset"]
    embeddings = query_only_benchmark["embeddings"]
    timing = query_only_benchmark.get("timing", {})
    inference_timing = measure_query_only_inference_time(
        data,
        embeddings,
        embedding_encode_avg_sec=timing.get("embedding_encode_avg_sec", 0.0),
    )

    elapsed = results_all[["dataset", "query_id", "pipeline", "elapsed_sec"]].copy()
    elapsed["query_id"] = elapsed["query_id"].astype(str)
    elapsed = elapsed.pivot_table(
        index=["dataset", "query_id"],
        columns="pipeline",
        values="elapsed_sec",
        aggfunc="first",
    ).reset_index()
    elapsed = elapsed.rename(columns={
        "BM25": "bm25_elapsed_sec",
        "BM25 + BGE-Reranker": "bm25_bge_elapsed_sec",
    })

    audit = query_only_benchmark["predictions"].copy()
    audit["query_id"] = audit["query_id"].astype(str)
    audit = audit.merge(elapsed, on=["dataset", "query_id"], how="left")
    audit = audit.merge(
        inference_timing[["model", "classifier_inference_avg_sec"]],
        on="model",
        how="left",
    )
    audit["reranker_extra_elapsed_sec"] = (
        audit["bm25_bge_elapsed_sec"] - audit["bm25_elapsed_sec"]
    ).clip(lower=0)
    audit["estimated_selective_elapsed_sec"] = (
        audit["bm25_elapsed_sec"]
        + audit["rerank_predicted"] * audit["reranker_extra_elapsed_sec"]
        + audit["classifier_inference_avg_sec"].fillna(0)
    )
    audit["time_saved_vs_always_rerank_sec"] = (
        audit["bm25_bge_elapsed_sec"] - audit["estimated_selective_elapsed_sec"]
    )
    audit["selected_delta_nDCG@10"] = audit["delta_nDCG@10"] * audit["rerank_predicted"]
    audit["actual_beneficial"] = audit["rerank_beneficial"].astype(int)
    audit["correct_decision"] = audit["rerank_predicted"].eq(audit["actual_beneficial"])
    audit["decision"] = np.where(audit["rerank_predicted"] == 1, "rerankear", "usar BM25")
    audit["actual_outcome"] = np.select(
        [audit["delta_nDCG@10"] > TARGET_THRESHOLD, audit["delta_nDCG@10"] < -TARGET_THRESHOLD],
        ["melhorou", "piorou"],
        default="empatou",
    )
    return audit, inference_timing


def summarize_query_only_policy(audit):
    rows = []
    for (model, evaluation), group in audit.groupby(["model", "evaluation"], sort=False):
        always_rerank_total = group["bm25_bge_elapsed_sec"].sum()
        selective_total = group["estimated_selective_elapsed_sec"].sum()
        rows.append({
            "model": model,
            "evaluation": evaluation,
            "queries": len(group),
            "positive_rate": group["actual_beneficial"].mean(),
            "accuracy": group["correct_decision"].mean(),
            "rerank_rate": group["rerank_predicted"].mean(),
            "mean_delta_always_rerank": group["delta_nDCG@10"].mean(),
            "mean_delta_selective_policy": group["selected_delta_nDCG@10"].mean(),
            "mean_time_bm25_sec": group["bm25_elapsed_sec"].mean(),
            "mean_time_always_rerank_sec": group["bm25_bge_elapsed_sec"].mean(),
            "mean_time_selective_policy_sec": group["estimated_selective_elapsed_sec"].mean(),
            "mean_classifier_inference_sec": group["classifier_inference_avg_sec"].mean(),
            "total_time_saved_vs_always_rerank_ratio": 1.0 - safe_ratio(selective_total, always_rerank_total),
        })
    return pd.DataFrame(rows)

def build_query_deltas(results_df):
    bm25_query = results_df[results_df["pipeline"] == "BM25"][[
        "dataset", "query_id", "ndcg", "mrr", "recall", "map"
    ]].rename(columns={
        "ndcg": "bm25_ndcg",
        "mrr": "bm25_mrr",
        "recall": "bm25_recall",
        "map": "bm25_map",
    })

    reranker_query = results_df[results_df["pipeline"] == "BM25 + BGE-Reranker"][[
        "dataset", "query_id", "ndcg", "mrr", "recall", "map"
    ]].rename(columns={
        "ndcg": "reranker_ndcg",
        "mrr": "reranker_mrr",
        "recall": "reranker_recall",
        "map": "reranker_map",
    })

    deltas = bm25_query.merge(reranker_query, on=["dataset", "query_id"])
    deltas["delta_nDCG@10"] = deltas["reranker_ndcg"] - deltas["bm25_ndcg"]
    deltas["delta_MRR@10"] = deltas["reranker_mrr"] - deltas["bm25_mrr"]
    deltas["delta_Recall@10"] = deltas["reranker_recall"] - deltas["bm25_recall"]
    deltas["delta_MAP@10"] = deltas["reranker_map"] - deltas["bm25_map"]
    return deltas

"""Carregamento tardio: analisar resultados não precisa inicializar o BGE."""
import torch
import time
from .progress import activity


class BGEReranker:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.setup_sec = 0.0

    def ensure_loaded(self):
        if self.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            start = time.perf_counter()
            options = {"revision": self.config["reranker_revision"]}
            with activity("Carregando BGE (download quando necessário)"):
                self.tokenizer = AutoTokenizer.from_pretrained(self.config["reranker_model"], **options)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config["reranker_model"], **options
                ).to(self.config["model_device"])
                self.model.eval()
            self.setup_sec = time.perf_counter() - start

    def __call__(self, query, candidates, documents):
        if not candidates:
            return []
        self.ensure_loaded()
        pairs = [(query, documents[item["doc_id"]]) for item in candidates]
        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.config["reranker_batch_size"]):
                inputs = self.tokenizer(
                    pairs[start:start + self.config["reranker_batch_size"]], padding=True,
                    truncation=True, max_length=self.config["reranker_max_length"], return_tensors="pt",
                ).to(self.model.device)
                scores.extend(self.model(**inputs).logits.reshape(-1).detach().cpu().tolist())
        return sorted(
            [{"doc_id": item["doc_id"], "score": float(score)} for item, score in zip(candidates, scores)],
            key=lambda item: item["score"], reverse=True,
        )

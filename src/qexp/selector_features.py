"""Atributos disponíveis antes da expansão. Este módulo não recebe relevâncias."""
import hashlib
import numpy as np

from .retrieval import tokenize
from .selection import qexp_rank_divergence

LEXICAL_FEATURES = ["query_words", "query_chars", "query_unique_ratio",
                    "relative_gap", "score_gap", "score_entropy", "lexical_overlap"]
BGE_FEATURES = ["rank_divergence", "bge_score_std", "bge_score_gap"]

LENGTH_FEATURES = LEXICAL_FEATURES[:3]
RARITY_FEATURES = ["query_idf_mean", "query_idf_max", "query_idf_std", "query_rare_ratio", "query_oov_ratio"]
FEATURE_SETS = {
    "lexical": LEXICAL_FEATURES,
    "bge": LEXICAL_FEATURES + BGE_FEATURES,
    "length": LENGTH_FEATURES,
    "rarity": RARITY_FEATURES,
    "length_rarity": LENGTH_FEATURES + RARITY_FEATURES,
    "bge_rarity": LEXICAL_FEATURES + BGE_FEATURES + RARITY_FEATURES,
}


def feature_names(name):
    if name not in FEATURE_SETS:
        raise ValueError(f"Conjunto de atributos desconhecido: {name}")
    return list(FEATURE_SETS[name])


def uses_bge(name):
    return bool(set(feature_names(name)) & set(BGE_FEATURES))


def uses_rarity(name):
    return bool(set(feature_names(name)) & set(RARITY_FEATURES))


class CorpusStatistics:
    """DF do corpus integral indexado; não recebe consultas de treino ou qrels.

    Rareza considera presença em documentos, não número de ocorrências. IDF
    normalizado: log((N+1)/(df+1))/log(N+1). Termos ausentes são separados.
    """
    def __init__(self, bm25):
        from collections import Counter
        import time
        start = time.perf_counter()
        self.documents = int(bm25.corpus_size)
        if self.documents < 1:
            raise ValueError("Corpus vazio para estatísticas de raridade.")
        self.df = Counter()
        for document in bm25.doc_freqs:
            self.df.update(document.keys())
        if len(bm25.doc_freqs) != self.documents:
            raise ValueError("Frequências documentais incompatíveis com o corpus.")
        self.setup_sec = time.perf_counter()-start

    def features(self, query):
        terms = set(tokenize(query))
        frequencies = [self.df[t] for t in terms if self.df.get(t, 0) > 0]
        idfs = np.log((self.documents+1)/(np.asarray(frequencies, dtype=float)+1))/np.log(self.documents+1)
        denominator = max(1, len(terms))
        return {"query_idf_mean": float(np.mean(idfs)) if len(idfs) else 0.,
                "query_idf_max": float(np.max(idfs)) if len(idfs) else 0.,
                "query_idf_std": float(np.std(idfs)) if len(idfs) else 0.,
                "query_rare_ratio": sum(df/self.documents <= .01 for df in frequencies)/denominator,
                "query_oov_ratio": (len(terms)-len(frequencies))/denominator}


def query_hash(query):
    normalized = " ".join(tokenize(query))
    return hashlib.sha256(normalized.encode()).hexdigest()


def lexical_features(query, signals):
    tokens = tokenize(query)
    return {"query_words": len(tokens), "query_chars": len(query),
            "query_unique_ratio": len(set(tokens)) / max(1, len(tokens)),
            **{name: float(signals[name]) for name in LEXICAL_FEATURES[3:]}}


def bge_features(initial, reranked, top_k):
    scores = [item["score"] for item in reranked[:top_k]]
    return {"rank_divergence": qexp_rank_divergence(initial, reranked, top_k),
            "bge_score_std": float(np.std(scores)) if scores else 0.0,
            "bge_score_gap": float(max(scores) - min(scores)) if scores else 0.0}

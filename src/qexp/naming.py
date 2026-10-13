"""IDs públicos de técnicas/seletores; tradução dos IDs históricos na fronteira."""
import copy
import re

PUBLIC_IDS = dict(zip(("A", "B", "C", "C_H4", "D", "F", "G", "K", "E", "E_RA", "L", "R"),
                      ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "s1", "s2", "s3", "s4")))
LEGACY_IDS = {v: k for k, v in PUBLIC_IDS.items()}


def public_id(value):
    return PUBLIC_IDS.get(value, value)


def internal_id(value):
    return LEGACY_IDS.get(value, value)


def public_text(value):
    """Rótulos dos relatórios; preserva o artigo português/inglês 'A'."""
    value = re.sub(r'\b(C_H4|E_RA|[BCDEFGKLR])\b', lambda m: public_id(m[0]), value)
    value = re.sub(r'(?m)(\|\s*)A(\s*\|)', r'\g<1>t1\2', value)
    return value.replace('A não usa BGE', 't1 não usa BGE')


def normalize_config(config):
    config = dict(config)
    if config.get("pipelines") is not None:
        config["pipelines"] = [internal_id(m) for m in config["pipelines"]]
    for key in ("selective_method", "heuristic_method"):
        if key in config:
            config[key] = internal_id(config[key])
    return config


def public_config(config):
    config = dict(config)
    if config.get("pipelines") is not None:
        config["pipelines"] = [public_id(m) for m in config["pipelines"]]
    for key in ("selective_method", "heuristic_method"):
        if key in config:
            config[key] = public_id(config[key])
    return config


def rename_table(frame, public=True):
    frame = frame.copy()
    convert = public_id if public else internal_id
    for column in ("pipeline_id", "selected_method", "method", "reference"):
        if column in frame:
            frame[column] = frame[column].map(convert)
    return frame


def rename_records(records, public=True):
    convert = public_id if public else internal_id
    return [{**a, **{k: convert(a[k]) for k in ("pipeline_id", "selected_method") if k in a}} for a in records]


def public_model(artifact):
    return translate_model(artifact, public_id)


def translate_model(artifact, convert=internal_id):
    a = copy.deepcopy(artifact)
    if "method" in a:
        a["method"] = convert(a["method"])
    if "methods" in a:
        a["methods"] = [convert(m) for m in a["methods"]]
    if "protocol" in a:
        a["protocol"]["method"] = convert(a["protocol"]["method"])
    if "protocols" in a:
        a["protocols"] = {convert(m): {**v, "method": convert(v["method"])} for m, v in a["protocols"].items()}
    for key in ("models", "component_models"):
        for entry in a.get(key, []):
            entry["method"] = convert(entry["method"])
    if "output_order" in a:
        a["output_order"] = [[convert(m), c] for m, c in a["output_order"]]
    return a

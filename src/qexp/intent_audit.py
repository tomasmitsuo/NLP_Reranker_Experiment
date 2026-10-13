"""Amostra cega para avaliação humana; nunca gera julgamentos semânticos."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import stable_seed

FIELDS = ['intent_preserved', 'entities_changed', 'negation_changed', 'relations_changed', 'unsupported_information']


def audit_development_generations(result, frame):
    """Somente consultas de desenvolvimento; não revisa saídas do teste para ajustar prompts."""
    queries = set(frame.loc[frame.split != 'test', 'query_text'])
    rows, seen = [], set()
    for g in result['generations']:
        if g['query'] not in queries or g['specification']['kind'] != 'keywords' or g['cache_key'] in seen:
            continue
        seen.add(g['cache_key'])
        rows.append({'query': g['query'], 'cache_key': g['cache_key'], 'status': g['status'],
                     'error': g.get('error', ''), 'raw': g['raw'],
                     'attempts': len(g.get('attempts', [g])), 'generation_sec': g['generation_sec']})
    return pd.DataFrame(rows, columns=['query', 'cache_key', 'status', 'error', 'raw', 'attempts', 'generation_sec'])


def export_intent_sample(result, frame, methods, output, *, per_stratum=3, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metrics = result['metrics'].set_index(['dataset', 'query_id', 'pipeline_id'])
    artifacts = {(a['dataset'], a['query_id'], a['pipeline_id']): a for a in result['artifacts']}
    records = []
    for r in frame[frame.split == 'test'].to_dict('records'):
        for method in methods:
            a = artifacts[(r['dataset'], r['query_id'], method)]
            row = metrics.loc[(r['dataset'], r['query_id'], method)]
            delta = r[f'delta_{method}']
            outcome = ('invalid' if row.generation_status == 'invalid_generation' else
                       'gain' if delta > 1e-12 else 'loss' if delta < -1e-12 else 'tie')
            content = a.get('controlled_expansion') or '\n'.join(a.get('expansions', []))
            if not content:
                content = '\n'.join(c['text'] for c in a.get('feedback_chunks', []))
            records.append({'dataset': r['dataset'], 'query_id': r['query_id'], 'method': method,
                            'outcome': outcome, 'delta': delta, 'query': r['query_text'],
                            'expansion': content, 'generation_status': row.generation_status})
    records = pd.DataFrame(records)
    sampled = []
    for key, group in records.groupby(['dataset', 'method', 'outcome'], sort=True):
        selected = group.sample(n=min(per_stratum, len(group)), random_state=stable_seed(seed, '/'.join(key))).copy()
        selected['population_stratum'] = len(group)
        selected['sample_stratum'] = len(selected)
        sampled.append(selected)
    sample = pd.concat(sampled, ignore_index=True)
    sample['sample_id'] = [hashlib.sha256(json.dumps([seed, r.dataset, r.query_id, r.method]).encode()).hexdigest()[:16]
                           for r in sample.itertuples()]
    sample = sample.sample(frac=1, random_state=seed)
    sample.to_csv(output/'key_private.csv', index=False)
    annotations = sample[['sample_id', 'query', 'expansion']].copy()
    for field in ['annotator', *FIELDS, 'notes']:
        annotations[field] = ''
    annotations.to_csv(output/'annotations.csv', index=False)
    (output/'README.md').write_text('''# Avaliação humana da intenção

Preencha uma cópia de annotations.csv por avaliador, sem consultar key_private.csv
antes da avaliação. A chave contém método, efeito e estrato para análise posterior.
Use um identificador em annotator e yes/no/unclear/not_applicable em cada campo.
Não use um LLM para preencher os rótulos que serão relatados como humanos.

- intent_preserved: a expansão mantém a necessidade original de informação?
- entities_changed: altera ou acrescenta entidade/população não sustentada?
- negation_changed: remove, acrescenta ou inverte negação?
- relations_changed: modifica relação, causalidade ou incerteza?
- unsupported_information: introduz especificidade ou afirmação não sustentada?

Texto de expansão vazio (falha de geração): not_applicable em todos os campos.
Registrar justificativa em notes para no/unclear em intent_preserved e para yes
nos campos de alteração. Não inferir preservação a partir de melhora no ranking.
Use dois avaliadores quando possível; mantenha as avaliações independentes.
A amostragem é estratificada por coleção, método e ganho/perda/empate/falha.
Taxas brutas da amostra não estimam taxas da coleção. O relatório agrega por
estrato e só calcula taxas ponderadas quando a anotação de cada avaliador está
completa. Falhas de geração são contadas separadamente, sem rótulo semântico.
''', encoding='utf-8')
    return annotations


def summarize_annotations(annotations, key, output):
    from sklearn.metrics import cohen_kappa_score
    annotations = pd.read_csv(annotations, dtype=str, keep_default_na=False)
    key = pd.read_csv(key, dtype={'sample_id': str, 'query_id': str}, keep_default_na=False)
    if annotations.duplicated(['sample_id', 'annotator']).any() or key.sample_id.duplicated().any():
        raise ValueError('Identificadores de avaliação duplicados.')
    if not set(annotations.sample_id) <= set(key.sample_id):
        raise ValueError('Amostras ausentes da chave.')
    active = annotations[annotations.annotator.str.strip() != ''].copy()
    if active.empty:
        raise ValueError('Nenhuma avaliação humana preenchida.')
    allowed = {'yes', 'no', 'unclear', 'not_applicable'}
    if any(not set(active[field]) <= allowed for field in FIELDS):
        raise ValueError('Preencha todos os campos com yes/no/unclear/not_applicable.')
    joined = active.drop(columns=['query', 'expansion'], errors='ignore').merge(key, on='sample_id', validate='many_to_one')
    empty = joined.expansion.str.strip().eq('')
    if ((joined.loc[empty, FIELDS] != 'not_applicable').any().any()
            or (joined.loc[~empty, FIELDS] == 'not_applicable').any().any()):
        raise ValueError('not_applicable é reservado às expansões vazias.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    strata = joined.groupby(['annotator', 'dataset', 'method', 'outcome', 'intent_preserved']).size().rename('annotations').reset_index()
    strata.to_csv(output/'strata.csv', index=False)
    totals = []
    for annotator, group in joined.groupby('annotator'):
        complete = set(group.sample_id) == set(key.sample_id)
        valid = group[group.expansion.str.strip() != '']
        weights = valid.population_stratum.astype(float)/valid.sample_stratum.astype(float)
        totals.append({'annotator': annotator, 'complete': complete, 'annotated': len(group),
                       'expected': len(key), 'empty_expansions': int(group.expansion.str.strip().eq('').sum()),
                       'weighted_preserved_rate_valid': float(np.average(valid.intent_preserved.eq('yes'), weights=weights))
                       if complete and len(valid) else None})
    pd.DataFrame(totals).to_csv(output/'completion.csv', index=False)
    agreement = []
    raters = sorted(joined.annotator.unique())
    for i, left in enumerate(raters):
        for right in raters[i+1:]:
            pair = joined[joined.annotator.isin([left, right]) & joined.expansion.str.strip().ne('')].pivot(
                index='sample_id', columns='annotator', values='intent_preserved').dropna()
            if not pair.empty:
                categories = set(pair[left]) | set(pair[right])
                agreement.append({'left': left, 'right': right, 'pairs': len(pair),
                                  'agreement': float((pair[left] == pair[right]).mean()),
                                  'cohen_kappa': float(cohen_kappa_score(pair[left], pair[right])) if len(categories) > 1 else None})
    pd.DataFrame(agreement, columns=['left', 'right', 'pairs', 'agreement', 'cohen_kappa']).to_csv(output/'agreement.csv', index=False)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(summarize_annotations(args.annotations, args.key, args.output))


if __name__ == '__main__':
    main()

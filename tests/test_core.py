import json
import re
from pathlib import Path

import pytest
from config import ANALYSIS_REFERENCE, ELABORATION_CONSTRAINTS
from core.generation_evaluator import evaluate_generation_output, load_constraint_map
from core.schemas import EMAIL_ANALYSIS_SCHEMA, validate_email_analysis
from core.pt_dialect import evaluate_pt_dialect, check_lexical_contrasts
from core.wf_fidelity import compute_word_fidelity_from_dialect
from core.writing_quality import evaluate_writing_quality
from core.scorecard import build_elaboration_scorecard, format_scorecard
from runner import score_analysis
from core.artifacts import ArtifactValidationError, build_manifest, load_manifest, validate_rows, write_manifest
from compare import validate_comparison_artifacts, _json_comparison
from core.statistics import paired_bootstrap


def test_reference_rows_validate():
    rows=[json.loads(x) for x in ANALYSIS_REFERENCE.read_text(encoding='utf8').splitlines() if x.strip()]
    assert len(rows)==20
    assert len({r['id'] for r in rows})==20
    for row in rows:
        ok, err=validate_email_analysis(row['ground_truth'])
        assert ok, err


def test_schema_is_strict_and_closed():
    assert set(EMAIL_ANALYSIS_SCHEMA['required']) == set(EMAIL_ANALYSIS_SCHEMA['properties'])
    assert EMAIL_ANALYSIS_SCHEMA['additionalProperties'] is False


def test_constraint_patterns_load():
    c=load_constraint_map(str(ELABORATION_CONSTRAINTS))
    assert len(c) == 20


def test_generation_na_when_no_criteria():
    out=evaluate_generation_output('Boa tarde.', {})
    assert out['instruction_adherence_score'] is None
    assert out['semantic_preservation_score'] is None


def test_generation_detects_source_placeholder():
    out=evaluate_generation_output('[Nome do Cliente], obrigado.', {}, '[Nome do Cliente]')
    assert any('bracket_placeholders' in x for x in out['adherence_details']['failed_items'])


def test_generation_detects_generic_placeholder_without_source_slot():
    out=evaluate_generation_output('Boa tarde, [Seu Nome].', {}, 'Responder ao cliente.')
    assert out['adherence_details']['placeholder_count'] == 1
    assert any('bracket_placeholders' in x for x in out['adherence_details']['failed_items'])


def test_generation_ignores_markdown_link_labels():
    out=evaluate_generation_output(
        'Consulte [a política](https://example.com/politica).', {}, 'Responder ao cliente.'
    )
    assert out['adherence_details']['placeholder_count'] == 0


def test_generation_mutations_are_detected():
    constraints = {
        'required_actions': [{'action_id': 'confirmar_envio', 'pattern': r'confirm\w* o envio'}],
        'required_facts': [{'fact_id': 'date', 'pattern': r'15 de outubro'}],
        'forbidden_changes': [{'forbidden_id': 'wrong_price', 'pattern': r'99,90 €'}],
    }
    good = evaluate_generation_output(
        'Confirmamos o envio em 15 de outubro. O valor permanece 89,90 €.', constraints
    )
    assert good['instruction_adherence_score'] == 100.0
    assert good['semantic_preservation_score'] == 100.0

    negated = evaluate_generation_output(
        'Não podemos confirmar o envio em 15 de outubro. O valor permanece 89,90 €.', constraints
    )
    assert negated['instruction_adherence_score'] < 100.0

    changed = evaluate_generation_output(
        'Confirmamos o envio em 16 de outubro. O valor é 99,90 €.', constraints
    )
    assert changed['semantic_preservation_score'] < 100.0
    assert changed['semantic_details']['missing_facts'] == ['date']
    assert changed['semantic_details']['forbidden_violations'][0]['forbidden_id'] == 'wrong_price'


def test_dialect_outputs_separate_signals():
    d=evaluate_pt_dialect('A equipa está a enviar o documento.', use_languagetool=False)
    assert 'ptpt_compliance_pct' in d
    assert 'ptbr_leakage_detected' in d
    assert 'euptvid_prob' in d



def test_lexical_contrasts_mask_structured_nonlexical_spans(monkeypatch):
    import core.pt_dialect as pt_dialect

    class LookupSpy:
        def __init__(self, dictionary):
            self.dictionary = dictionary
            self.lookups = []

        def lookup(self, word):
            self.lookups.append(word)
            return self.dictionary.lookup(word)

    ptbr = pt_dialect.get_ptbr_dictionary()
    ptpt = pt_dialect.get_ptpt_dictionary()
    if ptbr is None or ptpt is None:
        return

    ptbr_spy = LookupSpy(ptbr)
    ptpt_spy = LookupSpy(ptpt)
    monkeypatch.setattr(pt_dialect, "_HUNSPELL_PTBR", ptbr_spy)
    monkeypatch.setattr(pt_dialect, "_HUNSPELL_PTPT", ptpt_spy)

    text = "Antes https://example.com/path depois support@example.org fim."
    check_lexical_contrasts(text)

    protected_fragments = {
        fragment.lower()
        for token in pt_dialect.get_spacy_nlp()(text)
        if token.like_url or token.like_email
        for fragment in re.findall(
            r"\b[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)*\b",
            token.text,
            re.UNICODE,
        )
    }
    assert protected_fragments.isdisjoint(ptbr_spy.lookups)
    assert protected_fragments.isdisjoint(ptpt_spy.lookups)

def test_wf_does_not_depend_on_classifier_probability():
    d={'pt_dialect_score': 0.01, 'violations': [], 'ptpt_compliance_pct': 100, 'ptbr_leakage_detected': False}
    wf=compute_word_fidelity_from_dialect('A equipa enviou o documento.', d)
    assert wf['wf_score'] == 100.0


def test_wq_is_structural_and_deterministic():
    text='Boa tarde,\n\nA equipa está a enviar a fatura.\n\nCumprimentos,'
    a=evaluate_writing_quality(text, language='pt-PT', use_languagetool=False)
    b=evaluate_writing_quality(text, language='pt-PT', use_languagetool=False)
    assert a['writing_quality_score'] == b['writing_quality_score']
    assert 'grammar_error_count' in a and 'structural_issue_count' in a


def test_cost_unknown_is_not_zero():
    s=build_elaboration_scorecard([{'latency_ms':10,'cost_usd':None,'prompt_tokens':1,'completion_tokens':1}])
    assert s['cost_per_1k_emails_usd'] is None
    assert s['unknown_cost_samples'] == 1


def test_scorecard_reports_metric_denominators():
    s = build_elaboration_scorecard([
        {
            'status': 'success', 'target_lang': 'pt-pt',
            'instruction_adherence_pct': 80, 'semantic_preservation_pct': None,
            'ptpt_compliance_pct': 90, 'ptbr_leakage_detected': False,
            'wf_score': None, 'writing_quality_score': 70,
            'latency_ms': 10,
        },
        {'status': 'error', 'error': 'failed'},
    ])
    assert s['requested_samples'] == 2
    assert s['successful_samples'] == 1
    assert s['denominators']['instruction_adherence_pct'] == 1
    assert s['denominators']['semantic_preservation_pct'] == 0
    assert s['denominators']['ptpt_compliance_pct'] == 1
    assert 'semantic_preservation_pct' in s['unavailable_metrics']


def test_provider_reported_cost_is_used_when_catalog_cost_is_unknown():
    s = build_elaboration_scorecard([{
        'status': 'success', 'latency_ms': 10,
        'raw_response': {'usage': {
            'cost': 0,
            'cost_details': {'upstream_inference_cost': 0.000001},
        }},
    }])
    assert s['total_cost_usd'] == 0.000001
    assert s['unknown_cost_samples'] == 0


def test_dialect_metrics_remain_unavailable_without_dictionary_evidence():
    s = build_elaboration_scorecard([{
        'status': 'success', 'target_lang': 'pt-pt', 'latency_ms': 10,
        'dialect_evaluation': {
            'language_adherence_status': 'DEGRADED: pt_PT Hunspell dictionary unavailable',
            'violations': [],
        },
        'ptpt_compliance_pct': None,
        'ptbr_leakage_detected': None,
        'wf_score': None,
    }])
    assert s['ptpt_compliance_pct'] is None
    assert s['ptbr_leakage_pct'] is None
    assert s['wf_score'] is None


def test_failed_records_do_not_affect_quality_or_performance():
    s = build_elaboration_scorecard([
        {
            'id': 'ok', 'status': 'success', 'target_lang': 'pt-pt',
            'latency_ms': 100, 'started_at': 1, 'finished_at': 1.1,
            'cost_usd': 0.01, 'prompt_tokens': 10, 'completion_tokens': 20,
            'instruction_adherence_pct': 80, 'semantic_preservation_pct': 90,
            'euptvid_probability': 0.8, 'pt_dialect_score': 90,
            'ptpt_compliance_pct': 95, 'wf_score': 95,
            'writing_quality_score': 80,
        },
        {'id': 'failed', 'status': 'error', 'error': 'provider failed', 'latency_ms': 0},
    ])
    assert s['samples'] == 2
    assert s['successful_samples'] == 1
    assert s['failed_samples'] == 1
    assert s['instruction_adherence_pct'] == 80
    assert s['latency_p50_ms'] == 100
    # Wall-clock derived (finished_at - started_at = 0.1s for 1 successful
    # sample => 600/min); compared with tolerance because 1.1 - 1 is not exact
    # in binary floating point.
    assert s['throughput_emails_per_min'] == pytest.approx(600)


def test_euptvid_is_independent_of_dialect_score():
    s = build_elaboration_scorecard([{
        'status': 'success', 'target_lang': 'pt-pt',
        'latency_ms': 10, 'started_at': 1, 'finished_at': 1.01,
        'euptvid_probability': 0.75, 'pt_dialect_score': None,
    }])
    assert s['euptvid_probability'] == 0.75
    assert s['wf_score'] is None


def test_format_scorecard_uses_canonical_keys():
    formatted = format_scorecard({
        'schema_validity_pct': 95,
        'entity_f1': 0.8,
        'instruction_adherence_pct': 90,
        'semantic_preservation_pct': 85,
        'euptvid_probability': 0.7,
        'ptpt_compliance_pct': 88,
        'wf_score': 86,
        'local_writing_quality': 80,
    })
    assert formatted['understanding']['schema_validity_pct'] == 95
    assert formatted['understanding']['entity_f1'] == 0.8
    assert formatted['generation']['instruction_adherence_pct'] == 90
    assert formatted['ptpt']['euptvid_probability'] == 0.7
    assert formatted['writing']['local_writing_quality'] == 80


def test_analysis_reports_missing_failed_duplicate_and_unexpected_results():
    truth = [{'id': 'a', 'ground_truth': {'category': 'support'}}, {'id': 'b', 'ground_truth': {'category': 'support'}}]
    results = [
        {'id': 'a', 'status': 'error', 'error': 'failed'},
        {'id': 'a', 'status': 'success', 'parsed': {'category': 'support'}, 'is_valid_schema': True},
        {'id': 'extra', 'status': 'success', 'parsed': {}, 'is_valid_schema': False},
    ]
    scored = score_analysis(truth, results)
    # P1.1: 'a' is present but errored -> failed, not missing. Only 'b' is
    # genuinely absent. Previously 'a' incremented both counters, so 1 error
    # plus 1 absent row reported as 2 missing + 1 failed = 3 problems for 2
    # affected tasks.
    assert scored['missing_results'] == 1
    assert scored['failed_results'] == 1
    assert scored['duplicate_result_ids'] == ['a']
    assert scored['unexpected_result_ids'] == ['extra']


def test_artifact_validator_accepts_strict_rows():
    result = validate_rows(
        [{
            'id': 'task-1', 'model': 'model-a', 'status': 'success',
            'content': 'Resposta completa.', 'latency_ms': 10,
            'raw_response': {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Resposta completa.'}}]},
        }],
        kind='generation',
        expected_ids=['task-1'],
    )
    assert result['artifact_schema_version'] == '1.0'
    assert result['row_count'] == 1


def test_artifact_validator_rejects_truncated_and_refused_generation():
    rows = [{
        'id': 'task-1', 'model': 'model-a', 'status': 'success',
        'content': 'partial',
        'raw_response': {'choices': [{'finish_reason': 'length', 'message': {'content': 'partial'}}]},
    }]
    try:
        validate_rows(rows, kind='generation')
    except ArtifactValidationError as exc:
        assert 'finish_reason' in str(exc)
    else:
        raise AssertionError('truncated generation was accepted')

    rows[0]['raw_response'] = {
        'choices': [{'finish_reason': 'stop', 'message': {'refusal': 'cannot comply'}}],
    }
    try:
        validate_rows(rows, kind='generation')
    except ArtifactValidationError as exc:
        assert 'refusal' in str(exc)
    else:
        raise AssertionError('refused generation was accepted')


def test_artifact_validator_rejects_duplicate_and_missing_status():
    rows = [
        {'id': 'task-1', 'model': 'model-a', 'status': 'success'},
        {'id': 'task-1', 'model': 'model-a'},
    ]
    try:
        validate_rows(rows, kind='generation')
    except ArtifactValidationError as exc:
        message = str(exc)
    else:
        raise AssertionError('invalid artifact was accepted')
    assert 'duplicate ids' in message
    assert 'status must be success or error' in message


def test_artifact_validator_rejects_non_finite_numbers():
    try:
        validate_rows(
            [{'id': 'task-1', 'model': 'model-a', 'status': 'success', 'latency_ms': float('nan')}],
            kind='generation',
        )
    except ArtifactValidationError as exc:
        assert 'finite and non-negative' in str(exc)
    else:
        raise AssertionError('non-finite numeric field was accepted')


def test_manifest_binds_result_hash(tmp_path):
    result_path = tmp_path / 'run.jsonl'
    result_path.write_text('{"id":"task-1","model":"model-a","status":"success"}\n', encoding='utf-8')
    manifest = build_manifest(result_path, kind='generation', benchmark_version='0.1', model='model-a')
    write_manifest(result_path, manifest)
    loaded = load_manifest(result_path)
    assert loaded['result_sha256'] == manifest['result_sha256']
    result_path.write_text(result_path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    try:
        load_manifest(result_path)
    except ArtifactValidationError as exc:
        assert 'hash does not match' in str(exc)
    else:
        raise AssertionError('tampered artifact was accepted')


def test_compare_requires_explicit_legacy_mode(tmp_path):
    result_path = tmp_path / 'legacy.jsonl'
    result_path.write_text(
        '{"id":"elab_pt_01","model":"model-a","status":"success"}\n',
        encoding='utf-8',
    )
    try:
        validate_comparison_artifacts([result_path], kind='generation')
    except ArtifactValidationError as exc:
        assert 'missing manifest' in str(exc)
    else:
        raise AssertionError('legacy artifact was accepted without an explicit flag')
    metadata = validate_comparison_artifacts([result_path], kind='generation', allow_legacy=True)
    assert metadata[0]['legacy'] is True


def test_json_comparison_is_one_structured_document(tmp_path):
    first = {'artifact_provenance': 'manifest-backed', 'instruction_adherence_pct': 70}
    second = {'artifact_provenance': 'manifest-backed', 'instruction_adherence_pct': 80}
    first_path = tmp_path / 'first.jsonl'
    second_path = tmp_path / 'second.jsonl'
    first_path.write_text('{"model":"model-a"}\n', encoding='utf-8')
    second_path.write_text('{"model":"model-b"}\n', encoding='utf-8')
    report = _json_comparison(
        [first_path, second_path],
        [first, second],
        'generation',
    )
    assert len(report['artifacts']) == 2
    assert report['delta']['values']['instruction_adherence_pct'] == 10
    without_uncertainty = _json_comparison(
        [first_path, second_path], [first, second], 'generation',
        include_uncertainty=False,
    )
    assert 'pairwise_uncertainty' not in without_uncertainty


def test_paired_bootstrap_preserves_task_pairing():
    first = [
        {'id': 'a', 'status': 'success', 'instruction_adherence_pct': 50},
        {'id': 'b', 'status': 'success', 'instruction_adherence_pct': 100},
    ]
    second = [
        {'id': 'a', 'status': 'success', 'instruction_adherence_pct': 60},
        {'id': 'b', 'status': 'success', 'instruction_adherence_pct': 90},
    ]
    result = paired_bootstrap(first, second, metrics=('instruction_adherence_pct',), repetitions=100)
    metric = result['metrics']['instruction_adherence_pct']
    assert metric['n'] == 2
    assert metric['mean_difference'] == 0
    assert metric['wins'] == 1
    assert metric['losses'] == 1
    assert len(metric['ci95']) == 2


# ---------------------------------------------------------------------------
# P0 regression tests
# ---------------------------------------------------------------------------

def test_p0_1_throughput_uses_wall_clock_not_summed_latency():
    """Concurrent runs must not be penalised by summing per-request latency."""
    # 8 requests, 1000 ms each, all issued concurrently within the same second.
    records = [
        {
            'id': f'r{i}', 'status': 'success', 'latency_ms': 1000,
            'started_at': 100.0, 'finished_at': 101.0,
            'prompt_tokens': 10, 'completion_tokens': 10,
        }
        for i in range(8)
    ]
    s = build_elaboration_scorecard(records)
    # Wall clock is 1s for 8 emails => 480/min. The old summed-latency
    # fallback produced 60/min, i.e. 8x too low.
    assert s['throughput_emails_per_min'] == pytest.approx(480)
    assert s['tokens_per_second'] == pytest.approx(160)


def test_p0_1_throughput_unavailable_without_timestamps():
    """Legacy rows without timestamps report no throughput instead of a wrong one."""
    records = [
        {'id': f'r{i}', 'status': 'success', 'latency_ms': 1000}
        for i in range(8)
    ]
    s = build_elaboration_scorecard(records)
    assert s['throughput_emails_per_min'] is None
    assert s['tokens_per_second'] is None
    assert s['observed_wall_ms'] is None
    assert 'throughput_emails_per_min' in s['unavailable_metrics']
    assert 'tokens_per_second' in s['unavailable_metrics']


def test_p0_1_partial_timestamps_still_measure_wall_clock():
    """A failed row lacking timestamps must not discard the whole measurement."""
    records = [
        {'id': 'ok', 'status': 'success', 'latency_ms': 100,
         'started_at': 10.0, 'finished_at': 10.5},
        {'id': 'bad', 'status': 'error', 'error': 'boom'},
    ]
    s = build_elaboration_scorecard(records)
    assert s['observed_wall_ms'] == pytest.approx(500)
    assert s['throughput_emails_per_min'] == pytest.approx(120)


def test_p0_2_repaired_json_is_distinguished_from_clean_json():
    truth = [{'id': 'a', 'ground_truth': {}}, {'id': 'b', 'ground_truth': {}}]
    results = [
        {'id': 'a', 'status': 'success', 'parsed': {}, 'is_valid_schema': True,
         'parse_repaired': False},
        {'id': 'b', 'status': 'success', 'parsed': {}, 'is_valid_schema': True,
         'parse_repaired': True, 'raw_parse_error': 'Expecting value'},
    ]
    scored = score_analysis(truth, results)
    # Legacy headline metric is unchanged: both parse in the end.
    assert scored['schema_validity_pct'] == 100.0
    # But only one model output was valid without repair.
    assert scored['schema_validity_raw_pct'] == 50.0
    assert scored['schema_validity_repaired_pct'] == 50.0


def test_p0_4_unknown_cost_does_not_destroy_all_cost_data():
    records = [
        {'status': 'success', 'latency_ms': 10, 'cost_usd': 0.001}
        for _ in range(19)
    ]
    records.append({'status': 'success', 'latency_ms': 10, 'cost_usd': None})
    s = build_elaboration_scorecard(records)
    # Strict all-or-nothing total is preserved (a partial total is not a total).
    assert s['cost_per_1k_emails_usd'] is None
    assert s['unknown_cost_samples'] == 1
    # ...but the known-sample figure survives the single hiccup.
    assert s['known_cost_samples'] == 19
    assert s['cost_per_1k_emails_usd_known_only'] == pytest.approx(1.0)


def _corpus_texts():
    """Real PT-PT documents shipped with the benchmark (no invented sentences)."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data"
    sources = [
        ("analysis_reference.jsonl", ("email",)),
        ("wq_human_reference.jsonl", ("text", "content", "email")),
        ("calibration_dataset.jsonl", ("text", "content", "email")),
    ]
    out = []
    for name, keys in sources:
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            for key in keys:
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    out.append((str(obj.get("id")), value))
                    break
    return out


def test_p0_3_ter_detector_only_fires_on_labelled_ptbr_corpus_documents():
    """On the shipped PT-PT corpus the detector must not invent violations.

    Rather than asserting against hand-written sentences, this walks every
    reference document and requires that any PTBR_TER_HAVER hit belongs to a
    fixture explicitly labelled as a ter/haver PT-BR example.
    """
    from core.pt_dialect import check_lexicon_and_rules, get_spacy_nlp
    if get_spacy_nlp() is None:
        pytest.skip('spaCy pt_core_news_sm not installed')
    texts = _corpus_texts()
    assert texts, 'no reference corpus found'
    for doc_id, text in texts:
        hits = [
            i for i in check_lexicon_and_rules(text)
            if i['rule_id'] == 'PTBR_TER_HAVER'
        ]
        if hits and 'ter_haver' not in doc_id:
            raise AssertionError(
                f'false positive on unlabelled PT-PT document {doc_id}: {hits}'
            )


def test_p0_3_labelled_ter_haver_fixture_is_still_detected():
    """The fix must not silence the corpus fixture that is genuinely PT-BR."""
    from core.pt_dialect import check_lexicon_and_rules, get_spacy_nlp
    if get_spacy_nlp() is None:
        pytest.skip('spaCy pt_core_news_sm not installed')
    labelled = [t for i, t in _corpus_texts() if 'ter_haver' in i]
    assert labelled, 'expected a labelled ter/haver fixture in the corpus'
    for text in labelled:
        assert [
            i for i in check_lexicon_and_rules(text)
            if i['rule_id'] == 'PTBR_TER_HAVER'
        ]


def test_p0_3_prodrop_is_decided_by_morphology_not_a_word_list():
    """Pro-drop light-verb frames are generated combinatorially, not curated.

    Any bare or definite-marked object must be treated as a possessive /
    light-verb reading regardless of which noun fills the slot, so the rule
    scales to the lexicon instead of to an enumerated set of nouns.
    """
    import itertools
    from core.pt_dialect import check_lexicon_and_rules, get_spacy_nlp
    if get_spacy_nlp() is None:
        pytest.skip('spaCy pt_core_news_sm not installed')
    verbs = ['Tem', 'Tinha']
    objects = [
        'razão', 'tempo', 'medo', 'cuidado', 'paciência',
        'conhecimento', 'interesse', 'dificuldade',
        'a certeza', 'a informação', 'o direito', 'a possibilidade',
    ]
    for verb, obj in itertools.product(verbs, objects):
        sentence = f'{verb} {obj} para avançar.'
        assert not [
            i for i in check_lexicon_and_rules(sentence)
            if i['rule_id'] == 'PTBR_TER_HAVER'
        ], f'false positive on {sentence!r}'


def test_p0_3_existential_detection_generalises_over_indefinite_determiners():
    """Existential detection keys off Definite/PronType features, not lemmas."""
    import itertools
    from core.pt_dialect import check_lexicon_and_rules, get_spacy_nlp
    if get_spacy_nlp() is None:
        pytest.skip('spaCy pt_core_news_sm not installed')
    # Determiners spanning the indefinite article series and indefinite
    # quantifiers, each agreeing with its noun.
    frames = [
        ('um', 'problema'), ('uma', 'questão'), ('uns', 'documentos'),
        ('umas', 'questões'), ('vários', 'erros'), ('muita', 'gente'),
        ('alguns', 'clientes'), ('qualquer', 'falha'),
    ]
    for det, noun in frames:
        sentence = f'Tem {det} {noun} no sistema.'
        assert [
            i for i in check_lexicon_and_rules(sentence)
            if i['rule_id'] == 'PTBR_TER_HAVER'
        ], f'missed existential in {sentence!r}'


def test_p0_3_negated_proclisis_is_not_flagged():
    """'Não me diga' is correct PT-PT proclisis after a negation trigger."""
    from core.pt_dialect import check_lexicon_and_rules, get_spacy_nlp
    if get_spacy_nlp() is None:
        pytest.skip('spaCy pt_core_news_sm not installed')
    for sentence in ['Não me diga que o prazo mudou.', 'Nunca me disseram isso.']:
        flagged = [
            i for i in check_lexicon_and_rules(sentence)
            if i['rule_id'] == 'PTBR_PROCLISIS_START'
        ]
        assert not flagged, f'false positive on {sentence!r}'


def test_p0_3c_graded_compliance_key_present_on_every_return_branch():
    """Every exit path must expose the same keys (cf. P3.1 inconsistent keys).

    A key that silently disappears on some branches is indistinguishable from
    a genuine None to a `.get()` caller, which is exactly how the existing
    ptbr_violation_count inconsistency hides data.
    """
    from core.pt_dialect import evaluate_pt_dialect, get_spacy_nlp
    if get_spacy_nlp() is None:
        pytest.skip('spaCy pt_core_news_sm not installed')
    branch_inputs = [
        '',                                        # empty-text branch
        '   ',                                     # whitespace branch
        'This is an English email, entirely.',     # english-output branch
        'Bom dia, agradeço a sua mensagem.',       # normal scoring branch
    ]
    for text in branch_inputs:
        result = evaluate_pt_dialect(text, use_languagetool=False)
        assert 'ptpt_compliance_graded_pct' in result, (
            f'graded key missing for input {text!r}'
        )
        # The graded companion must agree with the binary metric on
        # availability: both known, or both unknown.
        assert (result['ptpt_compliance_pct'] is None) == (
            result['ptpt_compliance_graded_pct'] is None
        ), f'availability mismatch for {text!r}'


def test_p0_2_legacy_rows_without_parse_repaired_are_not_counted_as_clean():
    """Absence of `parse_repaired` is unknown, not 'was not repaired'."""
    truth = [{'id': 'a', 'ground_truth': {}}, {'id': 'b', 'ground_truth': {}}]
    legacy = [
        {'id': 'a', 'status': 'success', 'parsed': {}, 'is_valid_schema': True},
        {'id': 'b', 'status': 'success', 'parsed': {}, 'is_valid_schema': True},
    ]
    scored = score_analysis(truth, legacy)
    # The legacy headline metric still works...
    assert scored['schema_validity_pct'] == 100.0
    # ...but raw validity was never measured, so it must not claim 100%.
    assert scored['schema_validity_raw_pct'] is None
    assert scored['schema_validity_repaired_pct'] is None
    assert scored['denominators']['schema_validity_raw_pct'] == 0


def test_p0_4_all_costs_unknown_reports_none_not_zero():
    records = [{'status': 'success', 'latency_ms': 10, 'cost_usd': None}
               for _ in range(3)]
    s = build_elaboration_scorecard(records)
    assert s['cost_per_1k_emails_usd'] is None
    assert s['cost_per_1k_emails_usd_known_only'] is None
    assert s['known_cost_samples'] == 0
    assert s['unknown_cost_samples'] == 3


def test_p0_1_zero_span_does_not_produce_infinity():
    """Identical timestamps must not divide by zero."""
    records = [{'id': 'a', 'status': 'success', 'latency_ms': 0,
                'started_at': 5.0, 'finished_at': 5.0}]
    s = build_elaboration_scorecard(records)
    assert s['observed_wall_ms'] == 0.0
    assert s['throughput_emails_per_min'] is None
    assert s['tokens_per_second'] is None


def test_p0_1_malformed_timestamps_do_not_crash():
    records = [
        {'id': 'a', 'status': 'success', 'latency_ms': 10,
         'started_at': 'not-a-number', 'finished_at': None},
        {'id': 'b', 'status': 'success', 'latency_ms': 10,
         'started_at': 1.0, 'finished_at': 2.0},
    ]
    s = build_elaboration_scorecard(records)
    assert s['observed_wall_ms'] == pytest.approx(1000)


# ---------------------------------------------------------------------------
# P1 / P2 regression tests
# ---------------------------------------------------------------------------

def test_p2_4_non_dict_parsed_does_not_crash_scoring():
    """A JSON array/scalar parses fine but is not a dict; scoring must survive."""
    truth = [{'id': 'a', 'ground_truth': {'category': 'support'}},
             {'id': 'b', 'ground_truth': {'category': 'support'}}]
    for bad in ([{'category': 'support'}], 'a string', 42, True):
        rows = [
            {'id': 'a', 'status': 'success', 'parsed': bad,
             'is_valid_schema': False, 'parse_repaired': False},
            {'id': 'b', 'status': 'success', 'parsed': {'category': 'support'},
             'is_valid_schema': True, 'parse_repaired': False},
        ]
        scored = score_analysis(truth, rows)
        # Row is scored as a miss, NOT dropped: the denominator stays at 2 so a
        # malformed payload cannot improve accuracy by shrinking it.
        assert scored['denominators']['category_acc_pct'] == 2
        assert scored['category_acc_pct'] == 50.0


def test_p1_1_present_but_failed_row_is_not_also_counted_missing():
    """failed + missing + scored must reconcile against requested samples."""
    truth = [{'id': f't{i}', 'ground_truth': {'category': 'support'}}
             for i in range(20)]
    rows = [{'id': f't{i}', 'status': 'error', 'error': 'boom'} for i in range(3)]
    rows += [
        {'id': f't{i}', 'status': 'success', 'parsed': {'category': 'support'},
         'is_valid_schema': True, 'parse_repaired': False}
        for i in range(5, 20)
    ]
    scored = score_analysis(truth, rows)
    assert scored['failed_results'] == 3
    assert scored['missing_results'] == 2
    scored_count = scored['denominators']['category_acc_pct']
    assert scored_count == 15
    assert scored['failed_results'] + scored['missing_results'] + scored_count == \
        scored['requested_samples']


def test_p2_5_missing_wq_calibration_excludes_instead_of_crashing():
    """A missing frozen calibration artifact must not abort the WQ pipeline."""
    import importlib
    import shutil
    from pathlib import Path
    import core.writing_quality as wq_mod

    path = wq_mod._CALIBRATION_MODEL_PATH
    if not Path(path).is_file():
        pytest.skip('calibration artifact not present in this checkout')
    backup = Path(str(path) + '.p25bak')
    shutil.move(str(path), str(backup))
    try:
        importlib.reload(wq_mod)
        result = wq_mod.evaluate_writing_quality(
            'Bom dia, agradeco a sua mensagem. Com os melhores cumprimentos.',
            language='pt-PT', use_languagetool=False,
        )
        # No invented score, and the exclusion is explicit and machine-readable.
        assert result['writing_quality_score'] is None
        assert result['wq_scored'] is False
        assert result['wq_excluded'] is True
        assert 'calibration' in result['wq_excluded_reason'].lower()
    finally:
        shutil.move(str(backup), str(path))
        importlib.reload(wq_mod)

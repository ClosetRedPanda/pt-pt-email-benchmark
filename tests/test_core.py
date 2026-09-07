import json
from pathlib import Path
from config import ANALYSIS_REFERENCE, ELABORATION_CONSTRAINTS
from core.generation_evaluator import evaluate_generation_output, load_constraint_map
from core.schemas import EMAIL_ANALYSIS_SCHEMA, validate_email_analysis
from core.pt_dialect import evaluate_pt_dialect
from core.wf_fidelity import compute_word_fidelity_from_dialect
from core.writing_quality import evaluate_writing_quality
from core.scorecard import build_elaboration_scorecard, format_scorecard
from runner import score_analysis
from core.artifacts import ArtifactValidationError, build_manifest, load_manifest, validate_rows, write_manifest
from compare import validate_comparison_artifacts


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


def test_dialect_outputs_separate_signals():
    d=evaluate_pt_dialect('A equipa está a enviar o documento.', use_languagetool=False)
    assert 'ptpt_compliance_pct' in d
    assert 'ptbr_leakage_detected' in d
    assert 'euptvid_prob' in d


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
    assert s['throughput_emails_per_min'] == 600


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
    assert scored['missing_results'] == 2
    assert scored['failed_results'] == 1
    assert scored['duplicate_result_ids'] == ['a']
    assert scored['unexpected_result_ids'] == ['extra']


def test_artifact_validator_accepts_strict_rows():
    result = validate_rows(
        [{'id': 'task-1', 'model': 'model-a', 'status': 'success', 'latency_ms': 10}],
        kind='generation',
        expected_ids=['task-1'],
    )
    assert result['artifact_schema_version'] == '1.0'
    assert result['row_count'] == 1


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

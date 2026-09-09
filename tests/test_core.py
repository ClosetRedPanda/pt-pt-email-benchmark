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
from core.scorecard import build_elaboration_scorecard, format_scorecard, SCORECARD_SECTIONS
from runner import score_analysis
from core.artifacts import ArtifactValidationError, build_manifest, load_manifest, validate_rows, write_manifest
from compare import validate_comparison_artifacts, _json_comparison
from core.statistics import paired_bootstrap
from config import ELABORATION_PROMPTS, BENCHMARK_VERSION


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


def test_placeholder_detection_covers_locked_fixtures_generically():
    # REL-03: every Title-Case fixture in the roadmap's locked placeholder set
    # must be detected by the *generic* structural detector (no echo from the
    # source text), including slots with lowercase connectors such as
    # "[Nome do Cliente]". Note: all-lowercase "[inserir endereço]" is not
    # generically distinguishable from ordinary prose by structure alone and
    # remains only detectable when echoed from the source.
    fixtures = [
        '[Seu Nome]', '[Empresa]', '[Contato]', '[Nome do Cliente]',
        '[XXXX-XX-XX]', '[Nome Da Empresa]', '[Nome do Cliente e do Contato]',
    ]
    for fixture in fixtures:
        out = evaluate_generation_output(
            f'Boa tarde, {fixture}.\n\nCumprimentos', {}, 'Responder ao cliente.'
        )
        assert out['adherence_details']['placeholder_count'] >= 1, fixture


def test_placeholder_detection_ignores_citations_and_lowercase_prose():
    # Citations, years, all-lowercase bracketed prose and markdown links are
    # not template slots and must never be penalised.
    for text in [
        'Segundo o relatorio [1], o valor foi pago.',
        'Conforme [2024], o contrato renovou.',
        'Consulte [ver anexo] para detalhes.',
        'Consulte [a política](https://example.com/politica).',
    ]:
        out = evaluate_generation_output(text, {}, 'Responder ao cliente.')
        assert out['adherence_details']['placeholder_count'] == 0, text


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

def test_lexical_contrasts_treat_bare_protocol_scheme_tokens_as_dialect_neutral():
    # Issue 2 residual: bare protocol labels (https/http/ftp/...) are not words of
    # either dialect. The bundled PT-BR dictionary happens to list a few of them
    # while the PT-PT one does not, which used to report plain mentions such as
    # "via https" as PT-BR leakage. Scheme names are dialect-neutral by
    # construction, so no flag may be raised for them.
    assert check_lexical_contrasts("O acesso ao email via https ou http.") == []
    assert check_lexical_contrasts("O servidor antigo usa apenas ftp.") == []
    # A genuine lexical contrast in the same sentence is still reported.
    contexts = [
        issue["context"]
        for issue in check_lexical_contrasts("Envie o relatorio pela intranet via https.")
    ]
    assert "intranet" in contexts
    assert "https" not in contexts


def test_wq_defect_only_score_aggregated_alongside_calibrated_score():
    # Issue 3: writing_quality_score is calibration-capped below 100 for clean
    # text, which compresses model differences in the flawless range. The
    # defect-only counterpart keeps the full range; the scorecard must surface it.
    s = build_elaboration_scorecard([
        {
            'status': 'success', 'target_lang': 'pt-pt', 'latency_ms': 10,
            'instruction_adherence_pct': 90, 'semantic_preservation_pct': 90,
            'writing_quality_score': 91.5, 'wq_defect_only_score': 100.0,
        },
        {
            'status': 'success', 'target_lang': 'pt-pt', 'latency_ms': 10,
            'writing_quality_score': 70.0, 'wq_defect_only_score': 82.0,
        },
    ])
    assert s['local_writing_quality'] == pytest.approx(80.75)
    assert s['local_writing_quality_defect_only'] == pytest.approx(91.0)
    assert s['denominators']['local_writing_quality_defect_only'] == 2
    formatted = format_scorecard(s)
    assert formatted['writing']['local_writing_quality_defect_only'] == pytest.approx(91.0)
    assert 'local_writing_quality_defect_only' in formatted['writing']


def test_scorecard_defect_only_writing_metric_unavailable_without_evidence():
    s = build_elaboration_scorecard([{'status': 'success', 'latency_ms': 10}])
    assert s['local_writing_quality_defect_only'] is None
    assert 'local_writing_quality_defect_only' in s['unavailable_metrics']


def test_wq_defect_only_score_included_in_default_pairwise_metrics():
    from core.statistics import DEFAULT_METRICS
    assert "wq_defect_only_score" in DEFAULT_METRICS
    # Graded compliance is part of the default comparison set, ordered with its
    # binary sibling so pairwise reports show both.
    assert DEFAULT_METRICS.index("ptpt_compliance_pct") < DEFAULT_METRICS.index("ptpt_compliance_graded_pct")


def test_scorecard_surfaces_wq_evaluator_version():
    # REPRO-1: the summary must identify which writing-quality evaluator code
    # produced the rows (falls back to N/A for rows that predate the field).
    rows = [{
        'status': 'success',
        'writing_quality': {'wq_evaluator_version': 'task7-length-neutral-v1.1-school-scale'},
    }, {
        'status': 'success',
        'writing_quality': {},
    }]
    s = build_elaboration_scorecard(rows)
    assert s['wq_evaluator_version'] == 'task7-length-neutral-v1.1-school-scale'
    s2 = build_elaboration_scorecard([{'status': 'success', 'writing_quality': {}}])
    assert s2['wq_evaluator_version'] is None


def test_generation_reading_notes_explain_measurement_boundaries():
    # The pretty generation report appends notes that prevent over-reading the
    # headline metrics: pattern-coverage semantics, unused forbidden checks,
    # binary compliance, word-fidelity density, whole-run tokens/s.
    from compare import _generation_reading_notes
    notes = _generation_reading_notes({'wq_evaluator_version': 'v1'})
    joined = "\n".join(notes)
    assert 'pattern coverage' in joined
    assert 'no forbidden_changes' in joined
    assert 'binary per email' in joined
    assert 'density' in joined
    assert 'tokens/s' in joined
    assert 'v1' in joined


def test_wq_defect_only_score_is_the_primary_writing_metric():
    # Issue 3: primacy is structural, not just present — the defect-only score
    # leads the default pairwise metrics and every writing scorecard section,
    # with the calibrated score listed second everywhere.
    from core.statistics import DEFAULT_METRICS
    assert DEFAULT_METRICS.index("wq_defect_only_score") < DEFAULT_METRICS.index("writing_quality_score")
    assert SCORECARD_SECTIONS["writing"][0] == "local_writing_quality_defect_only"
    formatted = format_scorecard({
        "local_writing_quality_defect_only": 100.0,
        "local_writing_quality": 91.5,
    })
    assert list(formatted["writing"])[0] == "local_writing_quality_defect_only"


def test_full_rescore_recomputes_legacy_rows_with_current_evaluators():
    # compare.py --rescore must actually re-evaluate stored content. Legacy rows
    # keep frozen pre-fix values (URLs/protocol tokens flagged as PT-BR leaks, no
    # top-level wq_defect_only_score), so a rescore that only backfills None
    # fields is a silent no-op. After a full rescore those rows must show current
    # compliance/leakage and both writing-quality scores at the top level.
    from compare import _full_rescore_generation_records
    legacy = [
        {
            "id": "t1", "model": "m", "status": "success", "target_lang": "pt-pt",
            "content": "Boa tarde,\n\nJunte-se em https://meet.google.com/abc-defg-hij amanha.\n\nCumprimentos",
            "ptpt_compliance_pct": 0.0, "ptbr_leakage_detected": True, "wf_score": 98.0,
            "writing_quality_score": 91.5, "writing_quality": {"writing_quality_score": 91.5},
        },
        {
            "id": "t2", "model": "m", "status": "success", "target_lang": "pt-pt",
            "content": "Envie o relatorio pela intranet da empresa, por favor.",
            "ptpt_compliance_pct": 0.0, "ptbr_leakage_detected": True, "wf_score": 95.0,
        },
        {"id": "t3", "model": "m", "status": "error", "error": "provider boom", "content": ""},
    ]
    rows = _full_rescore_generation_records(legacy, use_languagetool=False)
    by_id = {row["id"]: row for row in rows}
    # URL row: no longer a leak under current masking; both WQ scores populated.
    assert by_id["t1"]["ptbr_leakage_detected"] is False
    assert by_id["t1"]["ptpt_compliance_pct"] == 100.0
    assert isinstance(by_id["t1"]["wq_defect_only_score"], (int, float))
    assert isinstance(by_id["t1"]["writing_quality_score"], (int, float))
    assert "wq_defect_only_score" in (by_id["t1"].get("writing_quality") or {})
    # Genuine lexical contrast must still be detected after a rescore.
    assert by_id["t2"]["ptbr_leakage_detected"] is True
    assert by_id["t2"]["ptpt_compliance_pct"] == 0.0
    # Failed rows are copied unchanged (no invented scores).
    assert by_id["t3"]["status"] == "error"
    assert "wq_defect_only_score" not in by_id["t3"]


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


def test_spelling_accepts_vocabulary_missing_from_one_pt_dictionary():
    # REL-06: spelling is orthography, not dialect. Words that are absent from
    # the bundled pt_PT dictionary but present in the pt_BR dictionary (or vice
    # versa) are correct Portuguese orthography and must not be reported as
    # spelling errors — the dialect layer judges dialect, the spelling layer
    # judges spelling. Real typos (absent from both) are still reported.
    text = ('Boa tarde,\n\nO estorno e o voucher foram registados na intranet '
            'e o minibar do hotel está incluído.\n\nCumprimentos')
    result = evaluate_writing_quality(text, language='pt-PT', use_languagetool=False)
    assert result['spelling_error_count'] == 0, result['wq_violations']


def test_spelling_still_flags_true_misspellings():
    # Diacritic-loss misspelling ("confirmacao" for "confirmação") remains a
    # spelling error even though both dictionaries were consulted.
    result = evaluate_writing_quality(
        'Boa tarde,\n\nA confirmacao do pedido segue em anexo.\n\nCumprimentos',
        language='pt-PT', use_languagetool=False,
    )
    assert result['spelling_error_count'] >= 1, result['wq_violations']


def test_task_echo_vocabulary_exempts_constraint_words_from_leakage():
    # REL-01/REL-05: when a task's own required-action pattern supplies a word
    # as an accepted answer (elab_pt_08: "reembolso|estorno"), the model is
    # expected to echo it, so it must not be flagged as a model-initiated PT-BR
    # leak. Without task context the strict dictionary contrast still applies.
    from core.generation_evaluator import constraint_echo_vocabulary
    spec = {'required_actions': [{'action_id': 'refund', 'pattern': 'reembolso|estorno'}]}
    echo = constraint_echo_vocabulary(spec)
    assert 'estorno' in echo and 'reembolso' in echo
    text = ('Boa tarde,\n\nVerificámos a cobrança duplicada e procedemos ao '
            'estorno do valor.\n\nCumprimentos')
    strict = evaluate_pt_dialect(text, use_languagetool=False)
    assert strict['ptbr_leakage_detected'] is True
    exempted = evaluate_pt_dialect(text, use_languagetool=False, echo_vocab=echo)
    assert exempted['ptbr_leakage_detected'] is False


def test_task_echo_exemption_is_structural_across_pattern_groups():
    from core.generation_evaluator import constraint_echo_vocabulary
    spec = {
        'required_facts': [{'fact_id': 'n', 'pattern': 'Marta Silva'}],
        'required_actions': [{'action_id': 'a', 'pattern': 'consultar a política na intranet|portal'}],
        'forbidden_changes': [{'change_id': 'w', 'pattern': 'valor de 240 euros'}],
    }
    echo = constraint_echo_vocabulary(spec)
    for word in ('silva', 'consultar', 'política', 'intranet', 'portal', 'valor', 'euros'):
        assert word in echo, word
    assert constraint_echo_vocabulary(None) == set()


def test_scorecard_reports_mixed_wq_evaluator_versions():
    # REL-11: rows from different evaluator versions must surface as "mixed",
    # not silently attributed to the first row's version.
    rows = [
        {'status': 'success', 'writing_quality': {'wq_evaluator_version': 'v1'}},
        {'status': 'success', 'writing_quality': {'wq_evaluator_version': 'v2'}},
    ]
    s = build_elaboration_scorecard(rows)
    assert s['wq_evaluator_version'] == 'mixed'
    same = build_elaboration_scorecard([
        {'status': 'success', 'writing_quality': {'wq_evaluator_version': 'v1'}},
        {'status': 'success', 'writing_quality': {'wq_evaluator_version': 'v1'}},
    ])
    assert same['wq_evaluator_version'] == 'v1'


def test_semantic_preservation_accepts_standard_pt_amount_format():
    # REL-02: elab_pt_02's amount fact must accept the standard PT-PT format
    # "1.240,00 €", not only the canonical "1,240 €" forms.
    from config import ELABORATION_CONSTRAINTS
    spec = json.loads(ELABORATION_CONSTRAINTS.read_text(encoding='utf-8'))['elab_pt_02']
    text = ('Boa tarde, Sra. Marta Silva,\n\nSegue hoje a fatura corrigida '
            'FT-4832 no valor de 1.240,00 €; pode contactar-nos para qualquer '
            'dúvida.\n\nCumprimentos')
    out = evaluate_generation_output(text, spec, source_text='')
    assert out['semantic_preservation_score'] == 100.0
    assert out['instruction_adherence_score'] == 100.0


def test_word_fidelity_standalone_honors_echo_vocab():
    # WF must not keep penalising task-echoed vocabulary once compliance no
    # longer does (internal consistency across metrics).
    from core.wf_fidelity import compute_word_fidelity
    from core.generation_evaluator import constraint_echo_vocabulary
    spec = {'required_actions': [{'action_id': 'refund', 'pattern': 'reembolso|estorno'}]}
    text = 'Boa tarde,\n\nProcedemos ao estorno do valor.\n\nCumprimentos'
    strict = compute_word_fidelity(text)
    exempt = compute_word_fidelity(text, echo_vocab=constraint_echo_vocabulary(spec))
    assert strict['wf_score'] < exempt['wf_score']
    assert exempt['wf_score'] == 100.0


def test_scorecard_grammar_unavailable_without_languagetool():
    # REL-07: without a LanguageTool backend, grammar counts are structural
    # zeros and must not be reported as a measured "0 errors per email".
    rows = [{
        'status': 'success',
        'writing_quality': {
            'languagetool_api_used': False,
            'grammar_error_count': 0,
            'spelling_error_count': 0,
        },
    } for _ in range(2)]
    s = build_elaboration_scorecard(rows)
    assert s['languagetool_available'] is False
    assert s['avg_grammar_errors_per_email'] is None
    assert 'grammar_errors_per_email' in s['unavailable_metrics']
    assert s['denominators']['grammar_errors_per_email'] == 0


def test_scorecard_grammar_reported_when_languagetool_responded():
    rows = [{
        'status': 'success',
        'writing_quality': {
            'languagetool_api_used': True,
            'grammar_error_count': count,
            'spelling_error_count': 0,
        },
    } for count in (2, 4)]
    s = build_elaboration_scorecard(rows)
    assert s['languagetool_available'] is True
    assert s['avg_grammar_errors_per_email'] == 3.0
    assert 'grammar_errors_per_email' not in s['unavailable_metrics']


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


def test_rescore_bypasses_the_version_gate_but_only_with_recomputation(tmp_path):
    """A stale-benchmark artifact must not be readable, but may be re-derived.

    The gate exists because criteria verdicts are frozen into each row. Refusing it
    outright, however, made `compare.py --rescore` unusable on exactly the artifacts
    it was meant to repair -- every run predating a criteria retarget. So the
    bypass is bound to the flag that actually recomputes those fields.
    """
    ids = [
        str(item["id"])
        for item in json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8")).get("prompts", [])
    ]
    result_path = tmp_path / "run.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for pid in ids:
            handle.write(json.dumps({
                "id": pid, "model": "model-a", "status": "success",
                "content": "Boa tarde.", "instruction_adherence_pct": 50.0,
                "semantic_preservation_pct": 50.0,
            }) + "\n")
    stale = build_manifest(
        result_path, kind="generation",
        benchmark_version="lean-0.0-definitely-stale", model="model-a",
    )
    write_manifest(result_path, stale)

    try:
        validate_comparison_artifacts([result_path], kind="generation", allow_legacy=True)
    except ArtifactValidationError as exc:
        assert "expected" in str(exc) and "--rescore" in str(exc)
    else:
        raise AssertionError("stale benchmark version was accepted without --rescore")

    # With --rescore the caller has committed to recomputing every score, so the
    # mismatch becomes a reported fact rather than a refusal.
    manifests = validate_comparison_artifacts(
        [result_path], kind="generation", allow_legacy=True, rescore=True,
    )
    assert manifests[0]["benchmark_version"] == "lean-0.0-definitely-stale"


def test_rescore_recomputes_criteria_instead_of_reading_frozen_values(tmp_path):
    """--rescore must refresh adherence, or it silently reports old criteria as new."""
    from compare import _refresh_criteria_records

    rows = [{
        "id": "elab_pt_01", "model": "model-a", "status": "success",
        "content": "Boa tarde, confirmamos o voucher de 15% para Rui Almeida na ordem PT-991.",
        "instruction_adherence_pct": 99.0, "semantic_preservation_pct": 99.0,
    }]
    refreshed = _refresh_criteria_records(rows)
    assert refreshed[0]["instruction_adherence_pct"] != 99.0, "frozen adherence was carried through"
    assert rows[0]["instruction_adherence_pct"] == 99.0, "input rows must not be mutated"


def test_rescore_output_never_overwrites_the_source_artifact(tmp_path):
    """A re-derivation lands in a new file; the record of what ran stays intact."""
    from compare import _persist_rescored

    ids = [
        str(item["id"])
        for item in json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8")).get("prompts", [])
    ]
    rows = [{"id": pid, "model": "model-a", "status": "success", "content": "Boa tarde."} for pid in ids]
    source = tmp_path / "run.jsonl"
    source.write_text(
        "\n".join(json.dumps({"id": pid, "model": "model-a", "status": "success", "content": "x"}) for pid in ids) + "\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    out_dir = tmp_path / "derived"
    _persist_rescored(source, rows, out_dir)
    assert source.read_bytes() == before, "source artifact was modified"
    assert (out_dir / "run.jsonl").is_file()
    stamped = json.loads((out_dir / "run.manifest.json").read_text(encoding="utf-8"))
    assert stamped["benchmark_version"] == BENCHMARK_VERSION
    assert stamped["parameters"]["criteria_refreshed"] is True
    try:
        _persist_rescored(source, rows, out_dir)
    except ArtifactValidationError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("re-derivation overwrote an existing artifact")


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


# ---------------------------------------------------------------------------
# P1.4 / P2.1 / P2.2 / P3 regression tests
# ---------------------------------------------------------------------------

def _corpus_emails():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data"
    out = []
    for name in ("wq_human_reference.jsonl", "analysis_reference.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                value = obj.get("email")
                if isinstance(value, str) and value.strip():
                    out.append(value)
    return out


def test_p1_4_sentence_initial_misspellings_are_detected():
    from core.writing_quality import detect_local_spelling_issues, _load_spellchecker
    if not _load_spellchecker('pt-PT'):
        pytest.skip('Hunspell pt-PT dictionary unavailable')
    for text in ('Tarrde boa.', 'Obrigadoo pela ajuda.', 'Reuniaoo marcada.'):
        assert detect_local_spelling_issues(text, 'pt-PT'), \
            f'sentence-initial misspelling missed in {text!r}'


def test_p1_4_no_new_false_positives_on_reference_corpus():
    """Checking capitalised tokens must not flag real PT-PT emails."""
    from core.writing_quality import detect_local_spelling_issues, _load_spellchecker
    if not _load_spellchecker('pt-PT'):
        pytest.skip('Hunspell pt-PT dictionary unavailable')
    emails = _corpus_emails()
    assert emails, 'no reference corpus found'
    for text in emails:
        for issue in detect_local_spelling_issues(text, 'pt-PT'):
            token = issue['message'].split("'")[1]
            assert token.islower() or token.lower() != token, token


def test_p1_4_capitalisation_rule_is_positional_not_lexical():
    """Mid-sentence capitals (proper nouns) and abbreviations stay unchecked."""
    from core.writing_quality import detect_local_spelling_issues, _load_spellchecker
    if not _load_spellchecker('pt-PT'):
        pytest.skip('Hunspell pt-PT dictionary unavailable')
    for text in ('Exmo. Senhor, junto envio o documento.',
                 'Ana Silva confirmou o pedido.',
                 'Enviei o FICHEIRO hoje.'):
        assert not detect_local_spelling_issues(text, 'pt-PT'), text


def test_p2_1_client_init_performs_no_network_io():
    import urllib.request
    from core.api_client import OpenRouterClient
    original = urllib.request.urlopen

    def _boom(*args, **kwargs):
        raise AssertionError('network I/O during __init__')

    urllib.request.urlopen = _boom
    try:
        OpenRouterClient(api_key='test-key')
        # An explicitly supplied map (including {}) must still be honoured.
        assert OpenRouterClient(api_key='k', pricing_map={}).pricing_map == {}
    finally:
        urllib.request.urlopen = original


def test_p2_2_rate_limiter_feedback_is_atomic():
    from core.api_client import RateLimiter
    limiter = RateLimiter(initial_rps=1.0, min_rps=0.1, max_rps=100.0)
    for _ in range(19):
        limiter.report_success()
    before = limiter.rps
    limiter.report_success()
    assert limiter.rps > before
    assert limiter._consecutive_success == 0
    limiter2 = RateLimiter(initial_rps=8.0, min_rps=0.1, max_rps=100.0)
    limiter2.report_429()
    assert limiter2.rps == 4.0
    assert limiter2._consecutive_success == 0


def test_p3_1_dialect_return_branches_expose_consistent_keys():
    from core.pt_dialect import evaluate_pt_dialect, get_spacy_nlp
    if get_spacy_nlp() is None:
        pytest.skip('spaCy pt_core_news_sm not installed')
    required = {'ptpt_compliance_pct', 'ptpt_compliance_graded_pct',
                'ptbr_violation_count', 'ptbr_candidate_count', 'violation_count'}
    for text in ('', '   ', 'This is an English email, entirely.',
                 'Bom dia, agradeco a sua mensagem.'):
        result = evaluate_pt_dialect(text, use_languagetool=False)
        assert required <= set(result), f'missing {required - set(result)} for {text!r}'


def test_p3_3_leakage_bootstrap_is_reported_as_a_rate():
    from core.statistics import paired_bootstrap
    first = [{'id': f't{i}', 'status': 'success', 'ptbr_leakage_detected': i < 2}
             for i in range(10)]
    second = [{'id': f't{i}', 'status': 'success', 'ptbr_leakage_detected': i < 6}
              for i in range(10)]
    report = paired_bootstrap(first, second, metrics=('ptbr_leakage_pct',),
                              repetitions=500)
    metric = report['metrics']['ptbr_leakage_pct']
    assert metric['n'] == 10
    # 20% -> 60% leakage is +40 percentage points, not +0.4.
    assert metric['mean_difference'] == pytest.approx(40.0)


def test_p3_7_constraint_patterns_have_no_duplicate_alternatives():
    import json
    from pathlib import Path
    from config import ELABORATION_CONSTRAINTS
    data = json.loads(Path(ELABORATION_CONSTRAINTS).read_text(encoding='utf-8'))

    def split_top_level(pattern):
        parts, buf, depth, in_class, escaped = [], '', 0, False, False
        for char in pattern:
            if escaped:
                buf += char
                escaped = False
                continue
            if char == '\\':
                buf += char
                escaped = True
                continue
            if char == '[' and not in_class:
                in_class = True
            elif char == ']' and in_class:
                in_class = False
            elif char == '(' and not in_class:
                depth += 1
            elif char == ')' and not in_class:
                depth -= 1
            if char == '|' and depth == 0 and not in_class:
                parts.append(buf)
                buf = ''
                continue
            buf += char
        parts.append(buf)
        return parts

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'pattern' and isinstance(value, str):
                    parts = split_top_level(value)
                    assert len(parts) == len(set(parts)), f'duplicate alternatives: {value}'
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)


# ---------------------------------------------------------------------------
# Managed external resources
# ---------------------------------------------------------------------------

def test_managed_resources_declare_pinned_revisions_and_digests():
    from core.resources import EUPTVID, HUNSPELL_RESOURCES, MANAGED_RESOURCES
    assert len(MANAGED_RESOURCES) == 5
    assert EUPTVID.relative_path == "models/model_quantized.ftz"
    assert {resource.relative_path for resource in HUNSPELL_RESOURCES} == {
        "docs/pt_PT.aff", "docs/pt_PT.dic", "docs/pt_BR.aff", "docs/pt_BR.dic",
    }
    for resource in MANAGED_RESOURCES.values():
        assert "/resolve/main/" not in resource.url
        assert "/master/" not in resource.url
        assert len(resource.sha256) == 64
        assert resource.size_bytes > 0


def test_managed_resource_rejects_corrupted_file(tmp_path, monkeypatch):
    """A file with the wrong digest must be reported, never used."""
    from core import resources
    bad = tmp_path / "model_quantized.ftz"
    bad.write_bytes(b"not the real model")
    monkeypatch.setattr(resources, "BASE_DIR", tmp_path.parent)
    fake = resources.ManagedResource(
        key="fake", relative_path=bad.name, url="https://example.invalid/x",
        sha256="0" * 64, size_bytes=1, description="test", license_note="test",
    )
    monkeypatch.setattr(type(fake), "path", property(lambda self: bad))
    problem = resources.verify(fake)
    assert problem is not None and "checksum mismatch" in problem
    assert not resources.is_available(fake)
    with pytest.raises(resources.ResourceUnavailable):
        resources.require(fake)


def test_missing_euptvid_yields_unavailable_not_fabricated(monkeypatch):
    """Absent model => unavailable signal, never an invented probability."""
    import core.pt_dialect as ptd
    from core import resources
    monkeypatch.setattr(resources, "verify", lambda r: "missing: test")
    monkeypatch.setattr(ptd, "_EUPTVID_MODEL", None)
    try:
        assert ptd.get_euptvid_model() is None
        signal = ptd.evaluate_euptvid_signal("Bom dia, agradeco a sua mensagem.")
        assert signal["available"] is False
        assert signal["ptpt_prob"] is None
        assert signal["label"] == "UNKNOWN"
    finally:
        ptd._EUPTVID_MODEL = None


def test_euptvid_classifies_when_resource_present():
    """When the managed model is installed the signal must actually work."""
    import core.pt_dialect as ptd
    from core.resources import EUPTVID, is_available
    if not is_available(EUPTVID):
        pytest.skip("EUPTVID resource not installed; run `python runner.py setup`")
    ptd._EUPTVID_MODEL = None
    ptpt = ptd.evaluate_euptvid_signal(
        "Bom dia, agradeco a sua mensagem e confirmo a reuniao de amanha."
    )
    ptbr = ptd.evaluate_euptvid_signal(
        "Oi, vou mandar o arquivo pra voce hoje de manha, ta bom?"
    )
    assert ptpt["available"] is True and ptbr["available"] is True
    # The PT-PT text must score higher on PT-PT than the PT-BR text does.
    assert ptpt["ptpt_prob"] > ptbr["ptpt_prob"]
    assert ptpt["label"] == "PT-PT"
    assert ptbr["label"] == "PT-BR"


def test_fasttext_numpy2_shim_is_applied_when_needed():
    """fastText 0.9.x + NumPy>=2 raises on every predict() without the shim."""
    import numpy as np
    from core.resources import EUPTVID, is_available
    if not is_available(EUPTVID):
        pytest.skip("EUPTVID resource not installed")
    if int(np.__version__.split(".", 1)[0]) < 2:
        pytest.skip("shim only needed on NumPy >= 2")
    import core.pt_dialect as ptd
    ptd._EUPTVID_MODEL = None
    model = ptd.get_euptvid_model()
    assert model is not None
    # Would raise ValueError("Unable to avoid copy...") unpatched.
    labels, probs = model.predict("Bom dia, tudo bem consigo?", k=-1)
    assert len(labels) == len(probs) > 0

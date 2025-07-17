from nltk.metrics.segmentation import pk, windowdiff
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)
from typing import Dict, List, Tuple

import tools

import os
from statistics import mean
from tools import extract_transcripts_to_file, extract_word_alignments_to_file

from aer import (
    calculate_metrics,
    parse_single_alignment,
    read_text,
)

def evaluate_segmentation_boundaries(
    alignment_hypotheses: Dict[str, dict],
    alignment_references: Dict[str, dict],
    on_punct: bool = False,
    aggregate: bool = False,
) -> List[float]:
    def get_boundaries(alignment: dict) -> Tuple[List[bool], List[bool]]:
        tokens_src = alignment["textA"].split()
        tokens_tgt = alignment["textB"].split()
        boundaries_src = [False] * (len(tokens_src) + 1)
        boundaries_tgt = [False] * (len(tokens_tgt) + 1)

        for pair in alignment["alignedPairs"]:
            if pair["parent"] is not None:
                continue

            src_span, tgt_span = pair["pair"]
            if src_span:
                if not on_punct or tokens_src[src_span[-1]] in ".!?":
                    boundaries_src[src_span[0]] = True
                    boundaries_src[src_span[-1] + 1] = True
            if tgt_span:
                if not on_punct or tokens_tgt[tgt_span[-1]] in ".!?":
                    boundaries_tgt[tgt_span[0]] = True
                    boundaries_tgt[tgt_span[-1] + 1] = True

        return boundaries_src, boundaries_tgt

    def binarize(bools: List[bool]) -> str:
        return "".join("1" if b else "0" for b in bools)

    def compute_metrics(ref: List[bool], hyp: List[bool]) -> Tuple[float, float, float, float, float, float, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(ref, hyp, average="binary", zero_division=0)
        acc = accuracy_score(ref, hyp)
        wd = windowdiff(binarize(ref), binarize(hyp), k=3)
        pk_val = pk(binarize(ref), binarize(hyp))
        kappa = cohen_kappa_score(ref, hyp)
        return acc, precision, recall, f1, wd, pk_val, kappa

    # Accumulate all boundaries
    all_ref_src, all_hyp_src = [], []
    all_ref_tgt, all_hyp_tgt = [], []

    for name, hyp_alignment in alignment_hypotheses.items():
        ref_alignment = alignment_references.get(name)
        if not ref_alignment:
            continue

        hyp_src, hyp_tgt = get_boundaries(hyp_alignment)
        ref_src, ref_tgt = get_boundaries(ref_alignment)

        all_hyp_src.extend(hyp_src)
        all_hyp_tgt.extend(hyp_tgt)
        all_ref_src.extend(ref_src)
        all_ref_tgt.extend(ref_tgt)

    # Compute metrics
    acc_s, prec_s, rec_s, f1_s, wd_s, pk_s, kappa_s = compute_metrics(all_ref_src, all_hyp_src)
    acc_t, prec_t, rec_t, f1_t, wd_t, pk_t, kappa_t = compute_metrics(all_ref_tgt, all_hyp_tgt)

    if aggregate:
        ref_all = all_ref_src + all_ref_tgt
        hyp_all = all_hyp_src + all_hyp_tgt
        acc_a, prec_a, rec_a, f1_a, wd_a, pk_a, kappa_a = compute_metrics(ref_all, hyp_all)
        return [
            100 * acc_a, 100 * prec_a, 100 * rec_a, 100 * f1_a,
            wd_a, pk_a, kappa_a
        ]
    else:
        return [
            100 * acc_s, 100 * prec_s, 100 * rec_s, 100 * f1_s,
            wd_s, pk_s, kappa_s,
            100 * acc_t, 100 * prec_t, 100 * rec_t, 100 * f1_t,
            wd_t, pk_t, kappa_t
        ]

def evaluate_exact_match(
    alignment_hypotheses: dict, alignment_references: dict, with_labels: bool = False
) -> float:
    exact_match = 0
    total_reference_pairs = 0

    for name, ref_alignment in alignment_references.items():
        hyp_alignment = alignment_hypotheses.get(name)
        if not hyp_alignment:
            continue

        ref_pairs = [
            pair for pair in ref_alignment["alignedPairs"] if pair["parent"] is None
        ]
        hyp_pairs = hyp_alignment["alignedPairs"]

        for ref_pair in ref_pairs:
            total_reference_pairs += 1
            for hyp_pair in hyp_pairs:
                if ref_pair["pair"] != hyp_pair["pair"]:
                    continue
                if with_labels and ref_pair["label"] != hyp_pair["label"]:
                    continue
                exact_match += 1
                break  # Only count one match per reference pair

    if total_reference_pairs == 0:
        return 0.0

    return 100 * exact_match / total_reference_pairs

def cohen_kappa_calculation(tp: int, fp: int, fn: int, tn: int) -> float:
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (
        (tp + fp) * (tn + fp) +
        (tp + fn) * (tn + fn)
    )

    return numerator / denominator if denominator != 0 else 0.0

def evaluate_segmentation_word_pairs(
    alignment_hypotheses: dict, alignment_references: dict
) -> tuple[float, float, float, float]:
    true_positives = false_positives = false_negatives = 0
    cohens_kappa = 0.0  # Default in case no data processed

    for name, hyp_alignment in alignment_hypotheses.items():
        ref_alignment = alignment_references.get(name)
        if not ref_alignment:
            continue

        src_count = len(hyp_alignment["textA"].split())
        tgt_count = len(hyp_alignment["textB"].split())

        def extract_links(pairs):
            return {
                f"{i}-{j}"
                for pair in pairs
                if pair["parent"] is None
                for i in pair["pair"][0]
                for j in pair["pair"][1]
            }

        pred_links = extract_links(hyp_alignment["alignedPairs"])
        ref_links = extract_links(ref_alignment["alignedPairs"])

        tp = len(pred_links & ref_links)
        fp = len(pred_links - ref_links)
        fn = len(ref_links - pred_links)
        tn = src_count * tgt_count - tp - fp - fn

        true_positives += tp
        false_positives += fp
        false_negatives += fn
        cohens_kappa = cohen_kappa_calculation(tp, fp, fn, tn)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1, cohens_kappa

def calculate_word_alignment_error_rate(
    source_path: str,
    target_path: str,
    source_transcript: str,
    target_transcript: str
) -> list[float]:
    sure_alignments = []
    possible_alignments = []
    hypothesis_alignments = []

    # Parameters — if you want them configurable, pass as function args
    clean_punctuation = False
    all_sure = False
    ignore_possible = False
    f_alpha = -1.0
    reverse_ref = False
    reverse_hyp = False

    # Read source and target texts
    source_tokens = read_text(source_transcript)
    target_tokens = read_text(target_transcript)

    assert len(source_tokens) == len(target_tokens), "Source and target length mismatch."
    assert not (clean_punctuation and not source_tokens), "Cannot clean punctuation without source/target text."

    # Parse reference alignments
    with open(source_path, 'r') as ref_file:
        for line in ref_file:
            sure = set()
            possible = set()
            for align_str in line.strip().split():
                is_sure = '-' in align_str
                align = parse_single_alignment(align_str, reverse_ref, one_indexed=True)

                if is_sure or all_sure:
                    sure.add(align)
                if is_sure or not ignore_possible:
                    possible.add(align)

            sure_alignments.append(sure)
            possible_alignments.append(possible)

    # Parse hypothesis alignments
    with open(target_path, 'r') as hyp_file:
        for line in hyp_file:
            alignment = {
                parse_single_alignment(a_str, reverse_hyp, one_indexed=True)
                for a_str in line.strip().split()
            }
            hypothesis_alignments.append(alignment)

    # Compute metrics
    precision, recall, aer, f_measure, *_ = calculate_metrics(
        sure_alignments,
        possible_alignments,
        hypothesis_alignments,
        f_alpha,
        source_tokens,
        target_tokens,
        clean_punctuation
    )

    return [aer, precision, recall, f_measure]


def get_word_alignment_eval(
    alignment_hypotheses: dict,
    alignment_references: dict,
    input_path: str,
    is_random: bool = False
) -> list[float]:
    ref_dir = "../tmp/ref"
    hyp_dir = "../tmp/hyp"
    os.makedirs(f"{ref_dir}/transcripts", exist_ok=True)
    os.makedirs(f"{hyp_dir}/transcripts", exist_ok=True)
    os.makedirs(f"{ref_dir}/links", exist_ok=True)
    os.makedirs(f"{hyp_dir}/links", exist_ok=True)

    extract_transcripts_to_file(alignment_references, f"{ref_dir}/transcripts")

    for name, ref_alignment in alignment_references.items():
        extract_word_alignments_to_file(ref_alignment, f"{ref_dir}/links/{name}.txt")

    for name, hyp_alignment in alignment_hypotheses.items():
        extract_word_alignments_to_file(hyp_alignment, f"{hyp_dir}/links/{name}.txt")

    aers, precisions, recalls, f1s = [], [], [], []

    for name in alignment_references:
        try:
            ref_link_path = f"{ref_dir}/links/{name}.txt"
            hyp_link_path = (
                os.path.join(input_path, f"{name}.txt")
                if is_random else
                f"{hyp_dir}/links/{name}.txt"
            )

            src_name, tgt_name = name.split("--")
            src_transcript = f"{ref_dir}/transcripts/{src_name}"
            tgt_transcript = f"{ref_dir}/transcripts/{tgt_name}"

            aer, precision, recall, f1 = calculate_word_alignment_error_rate(
                ref_link_path,
                hyp_link_path,
                src_transcript,
                tgt_transcript,
            )

            aers.append(aer)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            continue

    if not aers:
        return [0.0, 0.0]

    avg_precision = mean(precisions)
    avg_recall = mean(recalls)
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

    return [mean(aers), avg_f1]

def evaluate_span_labels(
    alignment_hypotheses: dict,
    alignment_references: dict,
    aggregate: bool = False,
) -> tuple[float, ...]:
    """
    Evaluate span-level labels for source and target texts in alignments.

    Args:
        alignment_hypotheses: Predicted alignments.
        alignment_references: Gold-standard alignments.
        aggregate: If True, return overall metrics for source+target combined.

    Returns:
        If aggregate=True:
            (accuracy_all, precision_all, recall_all, f1_all, kappa_all)
        If aggregate=False:
            (accuracy_source, precision_source, recall_source, f1_source, kappa_source,
             accuracy_target, precision_target, recall_target, f1_target, kappa_target)
        All scores except kappa are in percentage [0, 100].
    """
    ref_src_labels, ref_tgt_labels = [], []
    hyp_src_labels, hyp_tgt_labels = [], []

    for name in alignment_hypotheses:
        if name not in alignment_references:
            continue

        hyp_src, hyp_tgt = tools.get_labels(alignment_hypotheses[name])
        ref_src, ref_tgt = tools.get_labels(alignment_references[name])

        hyp_src_labels.extend(hyp_src)
        hyp_tgt_labels.extend(hyp_tgt)
        ref_src_labels.extend(ref_src)
        ref_tgt_labels.extend(ref_tgt)

    def compute_metrics(ref: list[str], hyp: list[str]) -> tuple[float, float, float, float, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            ref, hyp, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(ref, hyp)
        kappa = cohen_kappa_score(ref, hyp)
        return accuracy * 100, precision * 100, recall * 100, f1 * 100, kappa

    if aggregate:
        return compute_metrics(ref_src_labels + ref_tgt_labels, hyp_src_labels + hyp_tgt_labels)
    else:
        return compute_metrics(ref_src_labels, hyp_src_labels) + compute_metrics(ref_tgt_labels, hyp_tgt_labels)

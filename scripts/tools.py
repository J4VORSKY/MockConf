import os
import json
import random
from typing import Dict, List, Tuple, Any

def read_alignment_files(directory: str) -> Dict[str, Any]:
    """
    Reads all JSON alignment files in the given directory and returns a dictionary
    mapping from filename-based keys ("source--target") to the parsed JSON content.
    """
    annotations = {}
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            source, target = filename[:-5].split("--")
            annotations[f"{source}--{target}"] = data
        except (ValueError, json.JSONDecodeError):
            # Skip invalid JSON or improperly named files
            continue
    return annotations

def make_split(
    testset_file_names: List[str],
    devset_file_names: List[str],
    alignments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Splits the alignment data into devset and testset dictionaries
    based on the provided file name lists.
    """
    testset = {fname: alignments[fname] for fname in testset_file_names if fname in alignments}
    devset = {fname: alignments[fname] for fname in devset_file_names if fname in alignments}
    return devset, testset

def count_phrase_alignments(alignment_list: List[Dict[str, Any]]) -> int:
    """
    Counts how many alignment pairs are top-level (i.e., no parent).
    """
    return sum(1 for pair in alignment_list if pair.get("parent") is None)


def count_segments(alignment_list: List[Dict[str, Any]], side: str) -> int:
    """
    Counts the number of non-empty segments on the specified side ('source' or 'target')
    for top-level alignment pairs.
    """
    if side not in {"source", "target"}:
        raise ValueError("Side must be either 'source' or 'target'")
    
    index = 0 if side == "source" else 1
    return sum(
        1
        for pair in alignment_list
        if pair.get("parent") is None and pair["pair"][index]
    )

def create_random_baseline(reference_alignments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a random baseline alignment for each file in the reference_alignments.
    Only span alignment pairs are considered (not labeled as 'sure' or 'possible').
    """
    random_alignments = {}

    for filename, alignment in reference_alignments.items():
        aligned_pairs = alignment["alignedPairs"]
        textA_tokens = alignment["textA"].split()
        textB_tokens = alignment["textB"].split()

        # Filter non-sure/possible pairs
        def is_random_label(pair):
            return pair["label"] not in {"sure", "possible"}

        src_labels = [pair["label"] for pair in aligned_pairs if pair["pair"][0] and is_random_label(pair)]
        tgt_labels = [pair["label"] for pair in aligned_pairs if pair["pair"][1] and is_random_label(pair)]

        src_segment_count = len(src_labels)
        tgt_segment_count = len(tgt_labels)

        # Generate sorted random split points
        src_random_splits = sorted([0] + random.sample(range(1, len(textA_tokens) + 1), src_segment_count))
        tgt_random_splits = sorted([0] + random.sample(range(1, len(textB_tokens) + 1), tgt_segment_count))

        # Build label groupings
        src_additions = [("src", label) for label in src_labels if "addition" in label]
        tgt_additions = [("tgt", label) for label in tgt_labels if "addition" in label]
        src_rest = [("both", label) for label in src_labels if "addition" not in label]

        labels = src_additions + tgt_additions + src_rest
        random.shuffle(labels)

        random_alignment = {
            "textA": alignment["textA"],
            "textB": alignment["textB"],
            "selectedWordsA": [],
            "selectedWordsB": [],
            "selectedWordAlignment": None,
            "selectedPhraseAlignment": None,
            "alignedPairs": [],
        }

        src_index = tgt_index = 0
        for i, (side, label) in enumerate(labels):
            if side == "src":
                src_indices = list(range(src_random_splits[src_index], src_random_splits[src_index + 1]))
                tgt_indices = []
                src_index += 1
            elif side == "tgt":
                src_indices = []
                tgt_indices = list(range(tgt_random_splits[tgt_index], tgt_random_splits[tgt_index + 1]))
                tgt_index += 1
            else:  # both
                src_indices = list(range(src_random_splits[src_index], src_random_splits[src_index + 1]))
                tgt_indices = list(range(tgt_random_splits[tgt_index], tgt_random_splits[tgt_index + 1]))
                src_index += 1
                tgt_index += 1

            random_alignment["alignedPairs"].append({
                "pair": [src_indices, tgt_indices],
                "label": label,
                "labelIndex": 0,
                "note": label,
                "parent": None,
                "id": i,
            })

        random_alignments[filename] = random_alignment

    return random_alignments

def extract_word_alignments_to_file(alignment: dict, filepath: str) -> None:
    with open(filepath, "w") as file:
        alignments = []
        for pair in alignment["alignedPairs"]:
            if pair["parent"] is not None:
                src_idx = pair["pair"][0][0] + 1  # 1-based indexing
                tgt_idx = pair["pair"][1][0] + 1
                separator = "-" if pair["label"] == "sure" else "p"
                alignments.append(f"{src_idx}{separator}{tgt_idx}")
        file.write(" ".join(alignments) + "\n")

def extract_transcripts_to_file(alignments: dict, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    
    for alignment_name, alignment in alignments.items():
        source_name, target_name = alignment_name.split("--", maxsplit=1)
        
        with open(os.path.join(directory, source_name), "w") as src_file:
            src_file.write(alignment["textA"] + "\n")
        
        with open(os.path.join(directory, target_name), "w") as tgt_file:
            tgt_file.write(alignment["textB"] + "\n")

def get_labels(alignment: dict) -> tuple[list[str], list[str]]:
    default_label = "addition - extra information"
    source_tokens = alignment["textA"].strip().split()
    target_tokens = alignment["textB"].strip().split()

    source_labels = [default_label] * len(source_tokens)
    target_labels = [default_label] * len(target_tokens)

    for pair in alignment.get("alignedPairs", []):
        if pair.get("parent") is None:  # Phrase-level alignment
            label = pair.get("label", default_label)

            for idx in pair["pair"][0]:
                if 0 <= idx < len(source_labels):
                    source_labels[idx] = label

            for idx in pair["pair"][1]:
                if 0 <= idx < len(target_labels):
                    target_labels[idx] = label

    return source_labels, target_labels

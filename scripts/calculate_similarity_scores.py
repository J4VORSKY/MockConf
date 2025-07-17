import json
import argparse
import os
import tools
from constants import TESTSET_FILE_NAMES, DEVSET_FILE_NAMES
from collections import defaultdict
from typing import Dict, Any, Union

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# Load the multilingual sentence embedding model
model = SentenceTransformer("LaBSE")


def calculate_similarity_scores_for_labels(
    alignments: Dict[str, Any], extended: bool = False
) -> defaultdict[str, list[Union[float, tuple[float, str, str]]]]:
    """
    Calculate cosine similarity between aligned phrase pairs grouped by label.

    Args:
        alignments: Dictionary of alignment data.
        extended: If True, include source and target text in the result.

    Returns:
        A defaultdict mapping label -> list of cosine similarities (or tuples with text if extended=True).
    """
    similarity_scores = defaultdict(list)
    source_texts, target_texts, labels = [], [], []

    for alignment in alignments.values():
        src_tokens = alignment["textA"].split()
        tgt_tokens = alignment["textB"].split()

        for pair in alignment["alignedPairs"]:
            if pair["parent"] is not None:
                continue  # Skip nested alignments
            if len(pair["pair"][0]) == 0 or len(pair["pair"][1]) == 0:
                continue
            if "addition" in pair["label"]:
                continue

            src_phrase = " ".join(
                src_tokens[i] for i in pair["pair"][0] if i < len(src_tokens)
            )
            tgt_phrase = " ".join(
                tgt_tokens[i] for i in pair["pair"][1] if i < len(tgt_tokens)
            )

            source_texts.append(src_phrase)
            target_texts.append(tgt_phrase)
            labels.append(pair["label"])

    # Compute sentence embeddings
    all_embeddings = model.encode(source_texts + target_texts, convert_to_numpy=True)
    src_vecs = all_embeddings[:len(source_texts)]
    tgt_vecs = all_embeddings[len(source_texts):]

    for label, vec1, vec2, src, tgt in zip(labels, src_vecs, tgt_vecs, source_texts, target_texts):
        cos_sim = float(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))
        if extended:
            similarity_scores[label].append((cos_sim, src, tgt))
        else:
            similarity_scores[label].append(cos_sim)

    return similarity_scores


def main(input_path: str, output_path: str, extended: bool) -> None:
    reference_alignments = tools.read_alignment_files(input_path)
    reference_alignments_devset, _ = tools.make_split(
        TESTSET_FILE_NAMES, DEVSET_FILE_NAMES, reference_alignments
    )

    # Calculate similarity scores
    similarity_scores = calculate_similarity_scores_for_labels(reference_alignments_devset, extended=extended)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(similarity_scores, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate label-wise similarity scores from alignments.")
    parser.add_argument("input", help="Path to input alignment JSON file.")
    parser.add_argument("output", help="Path to output JSON file.")
    parser.add_argument("--extended", action="store_true", help="Include text spans in output.")

    args = parser.parse_args()

    main(args.input, args.output, args.extended)

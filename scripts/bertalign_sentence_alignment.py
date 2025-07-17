"""from ctypes import alignment
import json
import string
from bertalign import Bertalign
from numpy import add
from simalign import SentenceAligner

import argparse
import os

def get_subsegments(word_aligner, src_line, tgt_line, src_start_index, tgt_start_index) -> list:
    src_line = src_line.replace("...", ".")
    tgt_line = tgt_line.replace("...", ".")
    
    src_tokens = src_line.split()
    tgt_tokens = tgt_line.split()
    aligns, sim = word_aligner.get_word_aligns(src_line, tgt_line)

    assert len(list(aligns.values())) == 1, "Only one alignment is supported"

    punctuation_pairs = []
    for align in list(aligns.values())[0]:
        src_index, tgt_index = align
        src_word = src_tokens[src_index]
        tgt_word = tgt_tokens[tgt_index]
        if src_word in string.punctuation and tgt_word in string.punctuation:
            punctuation_pairs.append(align)

    subsegments = []
    last_src_index = src_start_index
    last_tgt_index = tgt_start_index
    for align in punctuation_pairs:
        src_index, tgt_index = align
        
        end_src_index = src_index + src_start_index + 1
        end_tgt_index = tgt_index + tgt_start_index + 1

        print(last_src_index, end_src_index, last_tgt_index, end_tgt_index)
        item_1 = list(range(last_src_index, end_src_index))
        item_2 = list(range(last_tgt_index, end_tgt_index))
        
        last_src_index = end_src_index
        last_tgt_index = end_tgt_index
        
        label = "translation"
        parent = None
        subsegments.append((item_1, item_2, label, parent))

    if len(subsegments) == 0:
        item_1 = list(range(src_start_index, src_start_index + len(src_tokens)))
        item_2 = list(range(tgt_start_index, tgt_start_index + len(tgt_tokens)))
        label = "translation"
        parent = None
        subsegments.append((item_1, item_2, label, parent))

    print(src_line)
    print(tgt_line)
    print(list(aligns.values())[0])
    print(subsegments)
    print()
    return subsegments
        
def add_word_alignment(word_aligner, alignments: dict, output_dir: str):
    for file, alignment in alignments.items():
        src_line = alignment["textA"]
        tgt_line = alignment["textB"]
        
        src_tokens = src_line.split()
        tgt_tokens = tgt_line.split()
        aligns, sim = word_aligner.get_word_aligns(src_line, tgt_line)

        assert len(list(aligns.values())) == 1, "Only one alignment is supported"

        word_links = []
        word_link_id = 10000
        for link in alignments[file]["alignedPairs"]:
            item_1, item_2 = link["pair"]

            if len(item_1) == 0 or len(item_2) == 0:
                continue

            src_index = item_1[0]
            tgt_index = item_2[0]
            src_length = len(item_1)
            tgt_length = len(item_2)

            src_text = " ".join(src_tokens[src_index:src_index + src_length])
            tgt_text = " ".join(tgt_tokens[tgt_index:tgt_index + tgt_length])

            aligns, sim = word_aligner.get_word_aligns(src_text, tgt_text)

            for align in list(aligns.values())[0]:
                i, j = align
                word_links.append({
                    "pair": [[i + src_index], [j + tgt_index]],
                    "label": "sure",
                    "labelIndex": 0,
                    "parent": link,
                    "note": "",
                    "id": word_link_id
                })

                word_link_id += 1

            print(src_text)
            print(tgt_text)
            print(list(aligns.values())[0])
            print()

        alignments[file]["alignedPairs"].extend(word_links)

        with open(f"{output_dir}/{file}", "w") as json_file:
            json.dump(alignments[file], json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--subsegments_word_alignment", action="store_true", default=False)
    parser.add_argument("--subsegments_bertalign", action="store_true", default=False)

    parser.add_argument("--max_align", type=int)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--window", type=int)
    parser.add_argument("--skip", type=float)
    parser.add_argument("--len_penalty", action="store_true", default=False)

    args, _ = parser.parse_known_args()

    word_aligner = SentenceAligner(
        model="xlmr",
        distortion=0.0,
        matching_methods="a",
        sentence_transformer=False,
    )

    if args.input_dir:
        files = os.listdir(args.input_dir)
        for file in files:
            print("Processing file: ", file)
            with open(os.path.join(args.input_dir, file), "r") as f:
                data = json.load(f)

                src = data["textA"]
                tgt = data["textB"]

                is_split = False
                if args.subsegments_bertalign:
                    src = src.replace(", ", ",\n")
                    src = src.replace(". ", ".\n")
                    src = src.replace("! ", "!\n")
                    src = src.replace("? ", "?\n")
                    src = src.replace("... ", "...\n")
                    src = src.strip()
                    
                    tgt = tgt.replace(", ", ",\n")
                    tgt = tgt.replace(". ", ".\n")
                    tgt = tgt.replace("! ", "!\n")
                    tgt = tgt.replace("? ", "?\n")
                    tgt = tgt.replace("... ", "...\n")
                    tgt = tgt.strip()
                    
                    is_split = True

                aligner = Bertalign(src, tgt, max_align=args.max_align, top_k=args.top_k, win=args.window, skip=args.skip, len_penalty=args.len_penalty, is_split=is_split)
                aligner.align_sents()

                src_cum_len = [0]
                tgt_cum_len = [0]

                for i in range(len(aligner.src_sents)):
                    last = src_cum_len[-1]
                    src_cum_len.append(len(aligner.src_sents[i].split()) + last)
                for i in range(len(aligner.tgt_sents)):
                    last = tgt_cum_len[-1]
                    tgt_cum_len.append(len(aligner.tgt_sents[i].split()) + last)

                sentence_alignment = {}
                sentence_alignment["textA"] = data["textA"]
                sentence_alignment["textB"] = data["textB"]
                
                sentence_alignment["selectedWordsA"] = []
                sentence_alignment["selectedWordsB"] = []
                sentence_alignment["selectedWordAlignment"] = None
                sentence_alignment["selectedPhraseAlignment"] = None

                sentence_alignment["alignedPairs"] = []

                print(len(src_cum_len))
                print(len(tgt_cum_len))
                
                index = 1
                
                label_to_indx = {
                    "translation": 0,
                    "addition - extra information": 4
                }

                with open(f"{args.output_dir}/{file}.max-align={args.max_align}.top-k={args.top_k}.window={args.window}.skip={args.skip}.len_penalty={args.len_penalty}.txt", "w") as txt_file:
                    for sen1, sen2 in aligner.result:
                        src_line = aligner._get_line(sen1, aligner.src_sents)
                        tgt_line = aligner._get_line(sen2, aligner.tgt_sents)
                        
                        label, desc = None, []
                        if len(sen1) == 0:
                            item_1 = []
                            item_2 = list(range(tgt_cum_len[sen2[0]], tgt_cum_len[sen2[-1] + 1]))
                            label = "addition - extra information"
                            desc = [(item_1, item_2, label, None)]
                        elif len(sen2) == 0:
                            item_1 = list(range(src_cum_len[sen1[0]], src_cum_len[sen1[-1] + 1]))
                            item_2 = []
                            label = "addition - extra information"
                            desc = [(item_1, item_2, label, None)]
                        else:
                            item_1 = list(range(src_cum_len[sen1[0]], src_cum_len[sen1[-1] + 1]))
                            item_2 = list(range(tgt_cum_len[sen2[0]], tgt_cum_len[sen2[-1] + 1]))
                            label = "translation"
                            
                            if args.subsegments_word_alignment:
                                desc = get_subsegments(word_aligner, src_line, tgt_line, src_start_index=src_cum_len[sen1[0]], tgt_start_index=tgt_cum_len[sen2[0]])
                            else:
                                desc = [(item_1, item_2, label, None)]
                            
                        
                        for item_1, item_2, label, parent in desc:
                            sentence_alignment["alignedPairs"].append({
                                "pair": [item_1, item_2],
                                "label": label,
                                "labelIndex": label_to_indx[label],
                                "parent": parent,
                                "note": "",
                                "id": index
                            })
                            index += 1
                        
                        txt_file.write(f"{src_line}\t{tgt_line}\n")

                    with open(f"{args.output_dir}/{file}", "w") as json_file:
                        json.dump(sentence_alignment, json_file, ensure_ascii=False, indent=4)

                    add_word_alignment(word_aligner, {file: sentence_alignment}, args.output_dir)"""

import os
import json
import string
import argparse
from collections import defaultdict

from simalign import SentenceAligner
from bertalign import Bertalign

def get_subsegments(word_aligner, src_line, tgt_line, src_start_idx, tgt_start_idx) -> list:
    """Get subsegment alignments based on punctuation anchors."""
    src_line = src_line.replace("...", ".")
    tgt_line = tgt_line.replace("...", ".")

    src_tokens = src_line.split()
    tgt_tokens = tgt_line.split()
    aligns, _ = word_aligner.get_word_aligns(src_line, tgt_line)

    assert len(aligns) == 1, "Only one alignment method is supported"

    punctuation_pairs = [
        align for align in list(aligns.values())[0]
        if src_tokens[align[0]] in string.punctuation and tgt_tokens[align[1]] in string.punctuation
    ]

    subsegments = []
    last_src, last_tgt = src_start_idx, tgt_start_idx

    for src_idx, tgt_idx in punctuation_pairs:
        end_src = src_idx + src_start_idx + 1
        end_tgt = tgt_idx + tgt_start_idx + 1

        item_1 = list(range(last_src, end_src))
        item_2 = list(range(last_tgt, end_tgt))
        subsegments.append((item_1, item_2, "translation", None))

        last_src, last_tgt = end_src, end_tgt

    if not subsegments:
        subsegments.append((
            list(range(src_start_idx, src_start_idx + len(src_tokens))),
            list(range(tgt_start_idx, tgt_start_idx + len(tgt_tokens))),
            "translation",
            None
        ))

    return subsegments

def add_word_alignments(word_aligner, alignments: dict, output_dir: str):
    """Add word-level alignments to existing sentence-level alignments."""
    for file, alignment in alignments.items():
        src_tokens = alignment["textA"].split()
        tgt_tokens = alignment["textB"].split()

        word_links = []
        word_link_id = 10000

        for link in alignment["alignedPairs"]:
            src_indices, tgt_indices = link["pair"]
            if not src_indices or not tgt_indices:
                continue

            src_start, tgt_start = src_indices[0], tgt_indices[0]
            src_len, tgt_len = len(src_indices), len(tgt_indices)

            src_text = " ".join(src_tokens[src_start:src_start + src_len])
            tgt_text = " ".join(tgt_tokens[tgt_start:tgt_start + tgt_len])

            aligns, _ = word_aligner.get_word_aligns(src_text, tgt_text)
            for i, j in aligns["inter"]:
                word_links.append({
                    "pair": [[i + src_start], [j + tgt_start]],
                    "label": "sure",
                    "labelIndex": 0,
                    "parent": link,
                    "note": "",
                    "id": word_link_id
                })
                word_link_id += 1

        alignment["alignedPairs"].extend(word_links)

        with open(os.path.join(output_dir, file), "w") as f_out:
            json.dump(alignment, f_out, ensure_ascii=False, indent=4)

def preprocess_for_bertalign(text):
    """Split text into lines based on sentence-ending punctuation."""
    for punct in [", ", ". ", "! ", "? ", "... "]:
        text = text.replace(punct, punct.strip() + "\n")
    return text.strip()

def process_file(file_path, args, word_aligner):
    """Main logic to process a single file and generate alignment."""
    with open(file_path, "r") as f:
        data = json.load(f)

    src, tgt = data["textA"], data["textB"]
    is_split = False

    if args.subsegments_bertalign:
        src = preprocess_for_bertalign(src)
        tgt = preprocess_for_bertalign(tgt)
        is_split = True

    aligner = Bertalign(
        src, tgt,
        max_align=args.max_align,
        top_k=args.top_k,
        win=args.window,
        skip=args.skip,
        len_penalty=args.len_penalty,
        is_split=is_split
    )
    aligner.align_sents()

    src_cum_len = [0]
    for sent in aligner.src_sents:
        src_cum_len.append(len(sent.split()) + src_cum_len[-1])

    tgt_cum_len = [0]
    for sent in aligner.tgt_sents:
        tgt_cum_len.append(len(sent.split()) + tgt_cum_len[-1])

    sentence_alignment = {
        "textA": data["textA"],
        "textB": data["textB"],
        "selectedWordsA": [],
        "selectedWordsB": [],
        "selectedWordAlignment": None,
        "selectedPhraseAlignment": None,
        "alignedPairs": []
    }

    label_to_index = {
        "translation": 0,
        "addition - extra information": 4
    }

    file_name = os.path.basename(file_path)
    out_txt_path = f"{args.output_dir}/{file_name}.max-align={args.max_align}.top-k={args.top_k}.window={args.window}.skip={args.skip}.len_penalty={args.len_penalty}.txt"

    with open(out_txt_path, "w") as txt_file:
        index = 1
        for sen1, sen2 in aligner.result:
            src_line = aligner._get_line(sen1, aligner.src_sents)
            tgt_line = aligner._get_line(sen2, aligner.tgt_sents)

            if not sen1:
                item_1 = []
                item_2 = list(range(tgt_cum_len[sen2[0]], tgt_cum_len[sen2[-1] + 1]))
                desc = [(item_1, item_2, "addition - extra information", None)]
            elif not sen2:
                item_1 = list(range(src_cum_len[sen1[0]], src_cum_len[sen1[-1] + 1]))
                item_2 = []
                desc = [(item_1, item_2, "addition - extra information", None)]
            else:
                item_1 = list(range(src_cum_len[sen1[0]], src_cum_len[sen1[-1] + 1]))
                item_2 = list(range(tgt_cum_len[sen2[0]], tgt_cum_len[sen2[-1] + 1]))
                if args.subsegments_word_alignment:
                    desc = get_subsegments(
                        word_aligner, src_line, tgt_line,
                        src_start_idx=src_cum_len[sen1[0]],
                        tgt_start_idx=tgt_cum_len[sen2[0]]
                    )
                else:
                    desc = [(item_1, item_2, "translation", None)]

            for item_1, item_2, label, parent in desc:
                sentence_alignment["alignedPairs"].append({
                    "pair": [item_1, item_2],
                    "label": label,
                    "labelIndex": label_to_index[label],
                    "parent": parent,
                    "note": "",
                    "id": index
                })
                index += 1

            txt_file.write(f"{src_line}\t{tgt_line}\n")

    output_json_path = os.path.join(args.output_dir, file_name)
    with open(output_json_path, "w") as out_file:
        json.dump(sentence_alignment, out_file, ensure_ascii=False, indent=4)

    add_word_alignments(word_aligner, {file_name: sentence_alignment}, args.output_dir)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subsegments_word_alignment", action="store_true")
    parser.add_argument("--subsegments_bertalign", action="store_true")
    parser.add_argument("--max_align", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--skip", type=float, required=True)
    parser.add_argument("--len_penalty", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    word_aligner = SentenceAligner(
        model="xlmr", distortion=0.0,
        matching_methods="a",
        sentence_transformer=False,
    )

    for file in os.listdir(args.input_dir):
        print(f"Processing: {file}")
        process_file(os.path.join(args.input_dir, file), args, word_aligner)

if __name__ == "__main__":
    main()

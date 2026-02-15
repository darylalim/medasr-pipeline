import re

import jiwer
import Levenshtein


def normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("</s>", "")
    s = re.sub(r"[^ a-z0-9']", " ", s)
    s = " ".join(s.split())
    return s


def compute_wer(ref_text: str, hyp_text: str) -> dict:
    normalized_ref = normalize(ref_text)
    normalized_hyp = normalize(hyp_text)
    measures = jiwer.process_words([normalized_ref], [normalized_hyp])
    return {
        "wer": measures.wer,
        "insertions": measures.insertions,
        "deletions": measures.deletions,
        "substitutions": measures.substitutions,
        "ref_tokens": len(normalized_ref.split()),
    }


def colored_diff(ref_text: str, hyp_text: str) -> str:
    ref_words = normalize(ref_text).split()
    hyp_words = normalize(hyp_text).split()
    edits = Levenshtein.editops(ref_words, hyp_words)

    r = 0
    parts = []

    for op, i, j in edits:
        if r < i:
            parts.append(" ".join(ref_words[r:i]))
        r = i

        if op == "replace":
            parts.append(f"\033[91m{{-{ref_words[i]}-}}\033[0m")
            parts.append(f"\033[92m{{+{hyp_words[j]}+}}\033[0m")
            r += 1
        elif op == "insert":
            parts.append(f"\033[92m{{+{hyp_words[j]}+}}\033[0m")
        elif op == "delete":
            parts.append(f"\033[91m{{-{ref_words[i]}-}}\033[0m")
            r += 1

    if r < len(ref_words):
        parts.append(" ".join(ref_words[r:]))

    return " ".join(parts)


def evaluate(ref_text: str, hyp_text: str) -> None:
    print("HYP:", hyp_text)
    wer_result = compute_wer(ref_text, hyp_text)
    print(
        f"WER: {wer_result['wer'] * 100:.2f}%: "
        f"insertions {wer_result['insertions']}, "
        f"deletions {wer_result['deletions']}, "
        f"substitutions {wer_result['substitutions']}, "
        f"ref tokens {wer_result['ref_tokens']}"
    )
    print(colored_diff(ref_text, hyp_text))

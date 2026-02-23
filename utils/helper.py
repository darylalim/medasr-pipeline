import html
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


def _diff_parts(ref_text, hyp_text, fmt_replace, fmt_insert, fmt_delete):
    ref_words = normalize(ref_text).split()
    hyp_words = normalize(hyp_text).split()
    edits = Levenshtein.editops(ref_words, hyp_words)

    r = 0
    parts: list[str] = []

    for op, i, j in edits:
        if r < i:
            parts.append(" ".join(ref_words[r:i]))
        r = i

        if op == "replace":
            parts.extend(fmt_replace(ref_words[i], hyp_words[j]))
            r += 1
        elif op == "insert":
            parts.append(fmt_insert(hyp_words[j]))
        elif op == "delete":
            parts.append(fmt_delete(ref_words[i]))
            r += 1

    if r < len(ref_words):
        parts.append(" ".join(ref_words[r:]))

    return " ".join(parts)


def colored_diff(ref_text: str, hyp_text: str) -> str:
    return _diff_parts(
        ref_text,
        hyp_text,
        fmt_replace=lambda ref, hyp: [
            f"\033[91m{{-{ref}-}}\033[0m",
            f"\033[92m{{+{hyp}+}}\033[0m",
        ],
        fmt_insert=lambda hyp: f"\033[92m{{+{hyp}+}}\033[0m",
        fmt_delete=lambda ref: f"\033[91m{{-{ref}-}}\033[0m",
    )


def html_diff(ref_text: str, hyp_text: str) -> str:
    return _diff_parts(
        ref_text,
        hyp_text,
        fmt_replace=lambda ref, hyp: [
            f'<span style="color:red;text-decoration:line-through">{html.escape(ref)}</span>',
            f'<span style="color:green;font-weight:bold">{html.escape(hyp)}</span>',
        ],
        fmt_insert=lambda hyp: (
            f'<span style="color:green;font-weight:bold">{html.escape(hyp)}</span>'
        ),
        fmt_delete=lambda ref: (
            f'<span style="color:red;text-decoration:line-through">{html.escape(ref)}</span>'
        ),
    )


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

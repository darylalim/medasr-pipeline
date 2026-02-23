from utils.helper import normalize, compute_wer, colored_diff, html_diff, evaluate


class TestNormalize:
    def test_lowercase(self):
        assert normalize("Hello World") == "hello world"

    def test_removes_eos_token(self):
        assert normalize("hello</s>") == "hello"

    def test_removes_special_characters(self):
        assert normalize("heart-rate: 80bpm!") == "heart rate 80bpm"

    def test_preserves_apostrophes(self):
        assert normalize("patient's") == "patient's"

    def test_collapses_whitespace(self):
        assert normalize("  multiple   spaces  ") == "multiple spaces"

    def test_empty_string(self):
        assert normalize("") == ""


class TestComputeWer:
    def test_identical_texts(self):
        result = compute_wer("hello world", "hello world")
        assert result["wer"] == 0.0
        assert result["insertions"] == 0
        assert result["deletions"] == 0
        assert result["substitutions"] == 0
        assert result["ref_tokens"] == 2

    def test_complete_mismatch(self):
        result = compute_wer("hello", "goodbye")
        assert result["wer"] == 1.0
        assert result["substitutions"] == 1

    def test_insertion(self):
        result = compute_wer("the cat", "the big cat")
        assert result["insertions"] == 1

    def test_deletion(self):
        result = compute_wer("the big cat", "the cat")
        assert result["deletions"] == 1

    def test_normalizes_before_comparison(self):
        result = compute_wer("Hello, World!", "hello world")
        assert result["wer"] == 0.0

    def test_ref_tokens_count(self):
        result = compute_wer("one two three four", "one two three four")
        assert result["ref_tokens"] == 4


class TestColoredDiff:
    def test_identical_texts(self):
        result = colored_diff("hello world", "hello world")
        assert "hello world" in result
        assert "\033[91m" not in result

    def test_substitution_markers(self):
        result = colored_diff("the cat sat", "the dog sat")
        assert "{-cat-}" in result
        assert "{+dog+}" in result

    def test_insertion_marker(self):
        result = colored_diff("the cat", "the big cat")
        assert "{+big+}" in result

    def test_deletion_marker(self):
        result = colored_diff("the big cat", "the cat")
        assert "{-big-}" in result

    def test_empty_strings(self):
        assert colored_diff("", "") == ""

    def test_empty_ref(self):
        result = colored_diff("", "hello")
        assert "{+hello+}" in result

    def test_empty_hyp(self):
        result = colored_diff("hello", "")
        assert "{-hello-}" in result


class TestHtmlDiff:
    def test_identical_texts(self):
        result = html_diff("hello world", "hello world")
        assert "<span" not in result
        assert "hello world" in result

    def test_substitution(self):
        result = html_diff("the cat sat", "the dog sat")
        assert (
            '<span style="color:red;text-decoration:line-through">cat</span>' in result
        )
        assert '<span style="color:green;font-weight:bold">dog</span>' in result

    def test_insertion(self):
        result = html_diff("the cat", "the big cat")
        assert '<span style="color:green;font-weight:bold">big</span>' in result

    def test_deletion(self):
        result = html_diff("the big cat", "the cat")
        assert (
            '<span style="color:red;text-decoration:line-through">big</span>' in result
        )

    def test_returns_string(self):
        result = html_diff("hello", "world")
        assert isinstance(result, str)

    def test_substitution_order(self):
        result = html_diff("the cat sat", "the dog sat")
        assert result.index("line-through") < result.index("font-weight:bold")

    def test_empty_strings(self):
        assert html_diff("", "") == ""

    def test_empty_ref(self):
        result = html_diff("", "hello")
        assert '<span style="color:green;font-weight:bold">hello</span>' in result

    def test_empty_hyp(self):
        result = html_diff("hello", "")
        assert (
            '<span style="color:red;text-decoration:line-through">hello</span>'
            in result
        )


class TestEvaluate:
    def test_prints_output(self, capsys):
        evaluate("hello world", "hello world")
        captured = capsys.readouterr()
        assert "HYP: hello world" in captured.out
        assert "WER: 0.00%" in captured.out

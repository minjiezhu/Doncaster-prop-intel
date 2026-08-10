from backend.app.eval.metrics import hit, reciprocal_rank


class TestHit:
    def test_true_when_expected_source_present(self) -> None:
        assert hit(["a.pdf", "b.pdf"], {"b.pdf"}) is True

    def test_false_when_no_expected_source_present(self) -> None:
        assert hit(["a.pdf", "c.pdf"], {"b.pdf"}) is False

    def test_false_on_empty_retrieval(self) -> None:
        assert hit([], {"b.pdf"}) is False


class TestReciprocalRank:
    def test_full_score_when_first_result_matches(self) -> None:
        assert reciprocal_rank(["b.pdf", "a.pdf"], {"b.pdf"}) == 1.0

    def test_half_score_when_second_result_matches(self) -> None:
        assert reciprocal_rank(["a.pdf", "b.pdf"], {"b.pdf"}) == 0.5

    def test_zero_when_no_match(self) -> None:
        assert reciprocal_rank(["a.pdf", "c.pdf"], {"b.pdf"}) == 0.0

    def test_matches_first_of_multiple_expected_sources(self) -> None:
        assert reciprocal_rank(["a.pdf", "b.pdf", "c.pdf"], {"c.pdf", "b.pdf"}) == 0.5

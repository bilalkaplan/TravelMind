import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import answer_validator


def load_retrieval_module_without_external_ml_dependencies():
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.intp = int
    fake_numpy.float32 = float
    fake_numpy.asarray = lambda values, dtype=None: tuple(values)
    fake_numpy.empty = lambda shape, dtype=None: tuple()

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = object

    module_name = "cmu_retrieve_focused_test"
    module_path = SRC / "cmu_retrieve.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "numpy": fake_numpy,
            "sentence_transformers": fake_sentence_transformers,
        },
    ):
        spec.loader.exec_module(module)
    return module


class ReviewRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retrieve = load_retrieval_module_without_external_ml_dependencies()

    def test_review_membership_index_is_cached_and_excludes_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "chunks.db"
            connection = sqlite3.connect(database_path)
            connection.execute(
                "CREATE TABLE chunks (id INTEGER PRIMARY KEY, chunk_type TEXT, metadata_json TEXT)"
            )
            rows = [
                (101, "cmu_review_group", {"hotel_id": "h1", "hotel_name": "Alpha Hotel", "chunk_type": "review_group"}),
                (102, "cmu_review_group", {"hotel_id": "h1", "hotel_name": "Alpha Hotel", "chunk_type": "review_group"}),
                (103, "cmu_hotel_profile", {"hotel_id": "h1", "hotel_name": "Alpha Hotel", "chunk_type": "hotel_profile"}),
                (104, "review", {"hotel_id": "h2", "hotel_name": "Beta Inn", "chunk_type": "review"}),
            ]
            connection.executemany(
                "INSERT INTO chunks (id, chunk_type, metadata_json) VALUES (?, ?, ?)",
                [(row_id, chunk_type, json.dumps(metadata)) for row_id, chunk_type, metadata in rows],
            )
            connection.commit()
            connection.close()

            self.retrieve.DB_PATH = database_path
            self.retrieve._cached_row_index = None
            self.retrieve.get_or_load_matrix = lambda: (None, [101, 102, 103, 104])

            first = self.retrieve.get_or_load_row_index()
            second = self.retrieve.get_or_load_row_index()

            self.assertIs(first, second)
            self.assertEqual(tuple(first["review_indices_by_hotel_id"]["h1"]), (0, 1))
            self.assertEqual(
                tuple(self.retrieve._resolve_hotel_review_indices("Alpha Hotel", first)),
                (0, 1),
            )
            self.assertEqual(
                tuple(self.retrieve._resolve_hotel_review_indices({"hotel_id": "h2"}, first)),
                (3,),
            )

    def test_search_returns_multiple_text_chunks_without_hotel_dedup(self):
        retrieve = self.retrieve
        retrieve.get_or_load_matrix = lambda: ("matrix", [101, 102, 999])
        retrieve.get_or_load_row_index = lambda: {
            "review_indices_by_hotel_id": {"h1": (0, 1)},
            "review_indices_by_hotel_name": {"alpha hotel": (0, 1)},
            "review_hotel_ids_by_name": {"alpha hotel": frozenset({"h1"})},
        }
        retrieve.get_or_load_embedding_model = lambda: types.SimpleNamespace(
            encode=lambda question, convert_to_numpy=True: "encoded-question"
        )

        observed = {}

        def rank(matrix, candidate_indices, query_embedding, k):
            observed["candidate_indices"] = tuple(candidate_indices)
            observed["question"] = query_embedding
            observed["k"] = k
            return [(1, 0.91), (0, 0.72)]

        retrieve._rank_review_indices_by_cosine = rank

        records = {
            101: {
                "chunk_id": "review-a",
                "chunk_type": "cmu_review_group",
                "text": "Rooms were clean and spacious.",
                "metadata": {"hotel_id": "h1", "hotel_name": "Alpha Hotel", "chunk_type": "review_group"},
            },
            102: {
                "chunk_id": "review-b",
                "chunk_type": "cmu_review_group",
                "text": "Some guests heard street noise.",
                "metadata": {"hotel_id": "h1", "hotel_name": "Alpha Hotel", "chunk_type": "review_group"},
            },
            999: {
                "chunk_id": "wrong-hotel",
                "chunk_type": "cmu_review_group",
                "text": "Must never be fetched.",
                "metadata": {"hotel_id": "h2", "hotel_name": "Beta Inn", "chunk_type": "review_group"},
            },
        }

        def fetch(row_ids):
            observed["fetched_ids"] = list(row_ids)
            return {row_id: records[row_id] for row_id in row_ids}

        retrieve.fetch_chunk_rows_by_id = fetch
        results = retrieve.search_reviews_for_hotel("h1", "What about the rooms?", k=8)

        self.assertEqual(observed["candidate_indices"], (0, 1))
        self.assertEqual(observed["fetched_ids"], [102, 101])
        self.assertEqual([result["chunk_id"] for result in results], ["review-b", "review-a"])
        self.assertEqual([result["metadata"]["hotel_id"] for result in results], ["h1", "h1"])
        self.assertTrue(all(result["text"] for result in results))


class AnswerValidatorTests(unittest.TestCase):
    def test_extract_final_answer_supports_stop_sequence_shape(self):
        self.assertEqual(
            answer_validator.extract_final_answer(
                "model preface <answer>Guests generally praise the rooms."
            ),
            "Guests generally praise the rooms.",
        )
        self.assertEqual(
            answer_validator.extract_final_answer(
                "<answer>Guests praise the service.</answer>ignored"
            ),
            "Guests praise the service.",
        )

    def test_extract_final_answer_only_drops_a_separate_meta_preamble(self):
        self.assertEqual(
            answer_validator.extract_final_answer(
                "Okay, I should answer from the reviews.\nThe rooms are usually described as clean."
            ),
            "The rooms are usually described as clean.",
        )
        same_line = "Okay, the rooms are usually described as clean."
        self.assertEqual(answer_validator.extract_final_answer(same_line), same_line)

    def test_warning_is_logged_but_does_not_replace_answer(self):
        answer = "This hotel has breakfast included."
        cards = [{"hotel_name": "Alpha Hotel", "amenities": {"breakfast": "UNKNOWN"}}]
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = answer_validator.validate_answer(
                answer, cards, "review_question", None, "en"
            )

        self.assertFalse(result["passed"])
        self.assertFalse(result["needs_fallback"])
        self.assertEqual(result["sanitized_answer"], answer)
        self.assertIn("breakfast_hallucination", result["warnings"])
        self.assertIn("answer_validator", stderr.getvalue())

    def test_evidence_prevents_a_claim_from_being_marked_hallucinated(self):
        answer = "This hotel has breakfast included."
        cards = [{"hotel_name": "Alpha Hotel", "amenities": {"breakfast": "UNKNOWN"}}]
        result = answer_validator.validate_answer(
            answer,
            cards,
            "review_question",
            None,
            "en",
            evidence_text="Review 4: This hotel has breakfast included.",
            allowed_hotel_names=["Alpha Hotel"],
        )

        self.assertTrue(result["passed"])
        self.assertFalse(result["needs_fallback"])
        self.assertEqual(result["sanitized_answer"], answer)

    def test_only_blocking_safety_categories_replace_the_answer(self):
        with mock.patch.object(
            answer_validator, "build_safe_fallback_answer", return_value="SAFE FALLBACK"
        ):
            reasoning = answer_validator.validate_answer(
                "Okay, the user is asking about rooms.", [], "review_question", None, "en"
            )
            price = answer_validator.validate_answer(
                "Book now for $120 per night.", [], "review_question", None, "en"
            )
            link = answer_validator.validate_answer(
                "See https://invented.example.com/hotel for details.",
                [],
                "review_question",
                None,
                "en",
            )

        for result in (reasoning, price, link):
            self.assertTrue(result["needs_fallback"])
            self.assertEqual(result["sanitized_answer"], "SAFE FALLBACK")

    def test_price_sentence_present_in_evidence_is_not_blocked(self):
        answer = "A guest mentioned a $20 parking fee."
        result = answer_validator.validate_answer(
            answer,
            [],
            "review_question",
            None,
            "en",
            evidence_text="Review text: A guest mentioned a $20 parking fee.",
        )

        self.assertFalse(result["needs_fallback"])
        self.assertEqual(result["sanitized_answer"], answer)


if __name__ == "__main__":
    unittest.main()

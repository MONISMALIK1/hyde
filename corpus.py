"""A small corpus + eval set for HyDE, built around *vocabulary mismatch*.

HyDE helps most when the user's wording shares few terms with the relevant
document — so retrieval on the raw query misses it — yet a hypothetical *answer*
would naturally use the document's vocabulary. Each eval question below is phrased
in everyday words while its gold document uses the technical term, which is exactly
the gap HyDE is meant to bridge (and what the offline tests demonstrate).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    question: str
    answer: str        # a substring expected in a correct answer
    gold_id: str       # the document that actually answers it
    note: str = ""


CORPUS: list[Document] = [
    Document("exceptions", "Exception handling",
             "In Python, runtime errors are handled with try and except blocks. "
             "Wrapping risky code in a try statement lets the program catch an "
             "exception and continue instead of terminating."),
    Document("binary_search", "Binary search",
             "Binary search locates an item in a sorted list by repeatedly halving "
             "the search interval, achieving O(log n) time complexity."),
    Document("inflation", "Inflation",
             "Inflation is the rate at which the general level of prices for goods "
             "and services rises, eroding purchasing power over time."),
    Document("mitochondria", "Mitochondria",
             "The mitochondrion is the organelle that produces most of a cell's ATP "
             "through aerobic respiration; it is often called the powerhouse of the cell."),
    Document("http_404", "HTTP 404",
             "An HTTP 404 status code indicates that the requested resource could not "
             "be found on the server."),
    Document("photosynthesis", "Photosynthesis",
             "Photosynthesis is the process by which green plants convert sunlight, "
             "water, and carbon dioxide into glucose and oxygen."),
    # off-topic distractors
    Document("great_wall", "Great Wall of China",
             "The Great Wall of China stretches thousands of kilometres and was built "
             "over many centuries."),
    Document("jupiter", "Jupiter",
             "Jupiter is the largest planet in the Solar System, a gas giant with a "
             "Great Red Spot."),
]


EVAL_QUESTIONS: list[QAExample] = [
    QAExample("How do I stop my program from crashing when something goes wrong?",
              "exception", "exceptions", note="everyday 'crash' -> try/except"),
    QAExample("What's a fast way to look something up in an ordered array?",
              "binary search", "binary_search", note="'fast lookup' -> binary search"),
    QAExample("Why does my money buy less than it used to?",
              "inflation", "inflation", note="'money buys less' -> inflation"),
    QAExample("Which part of a cell generates its energy?",
              "mitochond", "mitochondria", note="'energy' -> mitochondria"),
    QAExample("What does it mean when a web page says it can't be found?",
              "404", "http_404", note="'page not found' -> 404"),
    QAExample("How do plants make their own food from light?",
              "photosynthesis", "photosynthesis", note="'make food from light' -> photosynthesis"),
]


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def matches(predicted: str | None, example: QAExample) -> bool:
    if not predicted:
        return False
    return normalize(example.answer) in normalize(predicted)


__all__ = ["Document", "QAExample", "CORPUS", "EVAL_QUESTIONS", "matches", "normalize"]

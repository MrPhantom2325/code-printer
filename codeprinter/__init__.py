from __future__ import annotations

from typing import Any, Optional

from ._answers import ANSWERS, Answer

__all__ = ["ques", "register", "all_questions", "all_ques"]


def ques(question_number: int, *, default: Optional[Any] = None) -> Any:
    """Return the stored answer for the given question number.

    If the answer is callable it gets executed before returning the result.
    """
    if question_number in ANSWERS:
        result = ANSWERS[question_number]
        return result() if callable(result) else result
    if default is not None:
        return default
    raise KeyError(f"No answer registered for question {question_number}")


def register(question_number: int, answer: Answer) -> None:
    """Register or override an answer at runtime."""
    ANSWERS[question_number] = answer


def all_questions() -> dict[int, Any]:
    """Return a copy of the question-answer mapping."""
    resolved: dict[int, Any] = {}
    for key, value in ANSWERS.items():
        resolved[key] = value() if callable(value) else value
    return resolved


def all_ques() -> None:
    """Print all available question numbers."""
    question_numbers = sorted(ANSWERS.keys())
    if question_numbers:
        print("Available question numbers:", ", ".join(map(str, question_numbers)))
    else:
        print("No questions available.")


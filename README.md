# CodePrinter

`codeprinter` is a lightweight Python package for storing and retrieving numbered answers (for example, lab/program solutions) from a central mapping.

## Features

- Retrieve an answer by question number with `ques()`
- Register or override answers at runtime with `register()`
- Get all resolved answers with `all_questions()`
- Print available question numbers with `all_ques()`
- Supports both static values and callables (callables are executed on access)

## Installation

Install from PyPI:

```bash
pip install codeprinter2
```

For local development (editable install):

```bash
cd /path/to/code-printer
pip install -e .
```

## Quick Start

```python
import codeprinter as cp

# Get answer for question 1
print(cp.ques(1))

# Provide a fallback default instead of raising KeyError
print(cp.ques(999, default="Not available"))

# Register or override an answer at runtime
cp.register(10, "print('Hello from question 10')")
print(cp.ques(10))

# Register a callable answer (executed when requested)
cp.register(11, lambda: 6 * 7)
print(cp.ques(11))  # 42

# Fetch all answers (with callables resolved)
print(cp.all_questions())

# Print available question numbers
cp.all_ques()
```

## API

### `ques(question_number: int, *, default: Any | None = None) -> Any`
Returns the stored answer for `question_number`.

- If the stored answer is callable, it is executed and its return value is returned.
- If the question number is missing and `default` is provided, `default` is returned.
- If the question number is missing and `default` is not provided, a `KeyError` is raised.

### `register(question_number: int, answer: Answer) -> None`
Registers or overrides an answer in memory at runtime.

### `all_questions() -> dict[int, Any]`
Returns a new dictionary of all question-answer pairs with callable answers resolved.

### `all_ques() -> None`
Prints the list of available question numbers.

## Editing the built-in answer bank

Built-in entries are defined in:

- `codeprinter/_answers.py`

You can modify the `ANSWERS` mapping directly if you want package-level defaults to change.

## Requirements

- Python 3.8+

## License

MIT License © 2025 Shriyans Nayak

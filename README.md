# codeprinter

Tiny utility package for keeping your lab solutions in one place and importing them easily from anywhere in the world.

## 🌍 Global Installation

Once published to PyPI, you (or anyone) can install it globally using:

```bash
pip install codeprinter2
```

This will automatically install the latest version of the package from PyPI, so it can be used from any project or environment.

---

## 📦 Local Setup (for development)

If you’re developing the package locally, install it in editable mode:

```bash
cd /path/to/codeprinter
pip install -e .
```

This allows you to edit the code and immediately test changes without rebuilding.

---

## 🧠 Usage

After installation, simply import it anywhere in your Python code:

```python
import codeprinter as cp

print(cp.ques(1))
```

This will retrieve the answer or code snippet linked to that question number.

---

## 🛠️ Add or Update Answers

You can add new entries by editing `codeprinter/_answers.py`:

```python
ANSWERS = {
    1: "print('Hello world')",
    2: lambda: 42,
}
```

Each entry maps a **question number** to a **solution**, which can be:

* a string
* a dictionary
* or even a callable (function/lambda)

Callables are automatically executed when retrieved.

---

## 🧾 License

MIT License © 2025 Shriyans Nayak

---

## 📬 Author

**Shriyans Nayak**


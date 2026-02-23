# Contributing to PieceWise

Thank you for your interest in contributing to PieceWise.

## License Agreement — Please Read Before Contributing

PieceWise is released under the **Harsh Non-Commercial Attribution License (HNCAL) v1.0**.

**By submitting any contribution to this project — including pull requests, patches,
bug reports with proposed fixes, or documentation changes — you explicitly agree to
the following:**

1. Your contribution is submitted under the same terms as the HNCAL v1.0 license.
2. You grant the copyright holder (**Harsh Dwivedi**) a **perpetual, irrevocable, worldwide,
   royalty-free right** to use, modify, sublicense, and relicense your contribution
   under any license terms, including future commercial licenses, without further
   compensation or consent required.
3. You confirm that you have the legal right to make this grant and that your
   contribution does not infringe any third-party intellectual property rights.

If you do not agree to these terms, please do not submit contributions.

---

## How to Contribute

### Reporting Bugs
- Open a GitHub Issue with a clear title and description
- Include your OS, Python version, GPU type, and puzzle size
- Attach sample images if possible (or describe the failure mode)

### Suggesting Features
- Open a GitHub Issue tagged `enhancement`
- Describe the problem it solves and how it fits the existing pipeline

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature-name`
3. Follow the code style conventions below
4. Write or update tests for your change
5. Ensure all tests pass: `pytest tests/`
6. Submit a pull request against the `main` branch with a clear description

### Code Style
- Python: follow PEP 8, max line length 100
- All new modules must include a top-of-file docstring
- All new functions must include type annotations
- Structured logging via `structlog` — no bare `print()` statements in production code

### File Header
Every new Python file must include this header:

```python
# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.
```

---

## Questions?

Open a GitHub Discussion or contact: [harsh.dwivedi42@gmail.com]
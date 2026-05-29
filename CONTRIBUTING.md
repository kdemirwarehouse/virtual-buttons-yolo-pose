# Contributing

Thanks for your interest in contributing!

## Setup

```bash
git clone https://github.com/kdemirwarehouse/virtual-buttons-yolo-pose.git
cd virtual-buttons-yolo-pose
pip install -r requirements.txt
pip install ruff pytest
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting. Run before committing:

```bash
ruff check src/
```

## Submitting changes

1. Fork the repo and create a branch from `main`.
2. Make your changes with focused, atomic commits.
3. Open a pull request and describe what changed and why.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): short description
fix(scope): short description
docs: update README
chore: update deps
```

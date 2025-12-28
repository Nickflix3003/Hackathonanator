# Hackathonanator
AI Loop of Doom: a local Codex + GitHub PR loop driven by `todo.yaml`.

## What it does
- Reads tasks from `ai-pr-loop/todo.yaml`.
- Creates a branch per task (`ai/<task-id>`), runs Codex, verifies, and opens a PR.
- Tracks progress in `ai-pr-loop/state.json`.
- Writes prompts/outputs in `ai-pr-loop/logs/<run-id>/`.

## Setup
- Install deps: `pip install -r ai-pr-loop/requirements.txt`
- Ensure tooling: `git`, `codex`, and `gh` are available in PATH.

## Usage
- Dry run: `python ai-pr-loop/run_loop.py --dry-run`
- Run all tasks: `python ai-pr-loop/run_loop.py`
- Stop after one PR: `python ai-pr-loop/run_loop.py --max-prs 1`

## todo.yaml (high level)
- `project`: `name`, `base_branch`, `default_verify`
- `settings`: `max_attempts_per_task`, `pr_draft`, `allow_dirty_worktree`, `stop_on_failure`,
  `max_files_changed`, `max_lines_changed`, `max_run_minutes`, `max_task_minutes`, `codex_timeout_seconds`,
  `verify_timeout_seconds`, `max_verify_output_chars`, `max_prompt_chars`
- `tasks`: list of task objects with `id`, `title`, `description`, `acceptance_criteria`, `verify`

## Notes
- When `stop_on_failure` is true, the loop stops after a failed task or oversized change, leaving the branch for manual follow-up.
- When `stop_on_failure` is false, the loop will stash any uncommitted changes and continue; the stash ref is recorded in `state.json`.
- Set `max_files_changed` or `max_lines_changed` to detect oversized PRs.
- Use `max_task_minutes` and the timeout settings to cap long-running tasks or commands.
- Use `max_run_minutes` to cap the overall run time.
- Use `--rerun-failed` to re-run tasks not marked as `done`.
- Use `--reset-branch` to reset existing task branches to the base branch.

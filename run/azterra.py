import argparse
import subprocess
import sys
from pathlib import Path

# ---- CONFIG: change these once, then forget about paths ----

HACKATHONANATOR_ROOT = Path(__file__).resolve().parents[1]

REPO = Path(r"C:\Users\nickm\Desktop\Coding\Projects\Azterra\azterra")
TODO = REPO / ".ai-pr-loop" / "todo.yaml"

RUN_LOOP = HACKATHONANATOR_ROOT / "ai-pr-loop" / "run_loop.py"
TEMPLATE_TODO = HACKATHONANATOR_ROOT / "ai-pr-loop" / "todo.example.yaml"

# -----------------------------------------------------------


def ensure_todo_file(todo_path: Path, template_path: Path) -> None:
    if todo_path.exists():
        return
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    if template_path.exists():
        todo_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created todo file from template: {todo_path}")
    else:
        todo_path.write_text(
            "\n".join(
                [
                    "project:",
                    "  name: \"Azterra\"",
                    "  base_branch: \"main\"",
                    "  default_verify: []",
                    "",
                    "settings:",
                    "  max_attempts_per_task: 3",
                    "  pr_draft: true",
                    "  allow_dirty_worktree: false",
                    "  stop_on_failure: true",
                    "  max_files_changed: 0",
                    "  max_lines_changed: 0",
                    "  max_run_minutes: 0",
                    "  max_task_minutes: 15",
                    "  codex_timeout_seconds: 600",
                    "  verify_timeout_seconds: 600",
                    "  max_verify_output_chars: 12000",
                    "  max_prompt_chars: 30000",
                    "",
                    "tasks: []",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Created minimal todo file: {todo_path}")


def main():
    parser = argparse.ArgumentParser(description="Run PR-agent loop for Azterra")
    parser.add_argument("--repo", type=str, default=str(REPO), help="Path to Azterra repo")
    parser.add_argument("--todo", type=str, default="", help="Path to todo.yaml (default: <repo>/.ai-pr-loop/todo.yaml)")
    parser.add_argument("--run-loop", type=str, default=str(RUN_LOOP), help="Path to run_loop.py")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only")
    parser.add_argument("--max-prs", type=int, default=None, help="Limit number of PRs")
    parser.add_argument("--rerun-failed", action="store_true", help="Re-run tasks that are not marked done")
    parser.add_argument("--reset-branch", action="store_true", help="Reset existing task branches to base")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    todo_path = Path(args.todo).resolve() if args.todo else (repo_path / ".ai-pr-loop" / "todo.yaml")
    run_loop_path = Path(args.run_loop).resolve()
    template_path = run_loop_path.parent / "todo.example.yaml"
    if not template_path.exists():
        template_path = TEMPLATE_TODO

    ensure_todo_file(todo_path, template_path)

    cmd = [
        sys.executable,
        str(run_loop_path),
        "--repo", str(repo_path),
        "--todo", str(todo_path),
    ]

    if args.dry_run:
        cmd.append("--dry-run")

    if args.max_prs is not None:
        cmd.extend(["--max-prs", str(args.max_prs)])
    if args.rerun_failed:
        cmd.append("--rerun-failed")
    if args.reset_branch:
        cmd.append("--reset-branch")

    print("Running command:")
    print(" ".join(cmd))
    print()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Run loop failed with exit code {exc.returncode}.")
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()

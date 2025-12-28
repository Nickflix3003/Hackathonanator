import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


STATE_FILENAME = "state.json"


@dataclass
class ProjectConfig:
    name: str
    base_branch: str
    default_verify: List[str]


@dataclass
class SettingsConfig:
    max_attempts_per_task: int
    pr_draft: bool
    allow_dirty_worktree: bool


@dataclass
class Task:
    id: str
    title: str
    description: str
    acceptance_criteria: List[str]
    verify: List[str]


def sh(cmd: List[str], cwd: Path, check: bool = True, capture: bool = True) -> Tuple[int, str]:
    """Run a shell command and return (exit_code, combined_output)."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=os.environ.copy(),
    )
    out = proc.stdout or ""
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\n{out}")
    return proc.returncode, out


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:60] if len(s) > 60 else s


def load_todo(todo_path: Path) -> Tuple[ProjectConfig, SettingsConfig, List[Task]]:
    data = yaml.safe_load(todo_path.read_text(encoding="utf-8"))

    project = data.get("project", {})
    settings = data.get("settings", {})
    tasks_raw = data.get("tasks", [])

    proj = ProjectConfig(
        name=str(project.get("name", "Project")).strip(),
        base_branch=str(project.get("base_branch", "main")).strip(),
        default_verify=list(project.get("default_verify", [])),
    )

    sett = SettingsConfig(
        max_attempts_per_task=int(settings.get("max_attempts_per_task", 3)),
        pr_draft=bool(settings.get("pr_draft", True)),
        allow_dirty_worktree=bool(settings.get("allow_dirty_worktree", False)),
    )

    tasks: List[Task] = []
    for t in tasks_raw:
        tid = str(t["id"]).strip()
        title = str(t["title"]).strip()
        desc = str(t.get("description", "")).rstrip()
        ac = list(t.get("acceptance_criteria", []))
        verify = list(t.get("verify", proj.default_verify))
        if not verify:
            raise ValueError(f"Task '{tid}' has no verify commands and project.default_verify is empty.")
        tasks.append(Task(id=tid, title=title, description=desc, acceptance_criteria=ac, verify=verify))

    return proj, sett, tasks


def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "completed_tasks": {},  # task_id -> {branch, pr_url, attempts}
    }


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def ensure_clean_git(repo: Path, allow_dirty: bool) -> None:
    _, out = sh(["git", "status", "--porcelain"], cwd=repo, check=True)
    if out.strip() and not allow_dirty:
        raise RuntimeError(
            "Working tree is dirty. Commit/stash changes first, or set settings.allow_dirty_worktree=true."
        )


def current_branch(repo: Path) -> str:
    _, out = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, check=True)
    return out.strip()


def checkout_base(repo: Path, base_branch: str) -> None:
    sh(["git", "checkout", base_branch], cwd=repo, check=True)
    # Optional: pull latest if you want
    # sh(["git", "pull", "--ff-only"], cwd=repo, check=False)


def create_branch(repo: Path, branch: str) -> None:
    sh(["git", "checkout", "-b", branch], cwd=repo, check=True)


def commit_all(repo: Path, msg: str) -> None:
    sh(["git", "add", "-A"], cwd=repo, check=True)
    # If nothing changed, don't commit
    _, out = sh(["git", "status", "--porcelain"], cwd=repo, check=True)
    if not out.strip():
        raise RuntimeError("No changes to commit (working tree clean after Codex run).")
    sh(["git", "commit", "-m", msg], cwd=repo, check=True)


def push_branch(repo: Path, branch: str) -> None:
    sh(["git", "push", "-u", "origin", branch], cwd=repo, check=True)


def create_pr(repo: Path, title: str, body: str, base: str, head: str, draft: bool) -> str:
    cmd = ["gh", "pr", "create", "--title", title, "--body", body, "--base", base, "--head", head]
    if draft:
        cmd.append("--draft")
    _, out = sh(cmd, cwd=repo, check=True)
    # gh returns PR URL on success typically
    return out.strip()


def run_verify(repo: Path, commands: List[str]) -> Tuple[bool, str]:
    combined = []
    for c in commands:
        combined.append(f"$ {c}\n")
        # Run via shell to support "npm test && npm run build" etc.
        proc = subprocess.run(
            c,
            cwd=str(repo),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        combined.append(proc.stdout or "")
        if proc.returncode != 0:
            return False, "\n".join(combined)
    return True, "\n".join(combined)


def read_rules(repo: Path) -> str:
    rules_path = repo / "ai-pr-loop" / "AI_RULES.md"
    if rules_path.exists():
        return rules_path.read_text(encoding="utf-8").strip()
    return ""


def codex_exec(repo: Path, prompt: str, json_output: bool = True) -> str:
    """
    Calls Codex CLI. Assumes 'codex' is installed and authenticated.
    Uses 'codex exec' so this script can run non-interactively.
    """
    cmd = ["codex", "exec"]
    if json_output:
        cmd.append("--json")
    cmd.append(prompt)

    # We capture output; Codex will print logs + maybe JSON.
    _, out = sh(cmd, cwd=repo, check=True)
    return out


def build_task_prompt(project: ProjectConfig, task: Task, rules: str) -> str:
    ac_lines = "\n".join([f"- {x}" for x in task.acceptance_criteria]) if task.acceptance_criteria else "- (none provided)"
    verify_lines = "\n".join([f"- `{x}`" for x in task.verify])

    return f"""You are Codex, a coding worker inside a repo. Implement the task below.

Project: {project.name}

Rules:
{rules}

Task ID: {task.id}
Title: {task.title}

Description:
{task.description}

Acceptance criteria:
{ac_lines}

Verify commands (must pass after your changes):
{verify_lines}

Instructions:
- Make the smallest correct change set.
- Do not refactor unrelated code.
- If you add a new feature, update any relevant docs or UI text.
- After edits, ensure the project builds/tests per the verify commands (I will run them).
Return a short summary of what you changed and which files were modified.
"""


def build_fix_prompt(task: Task, rules: str, verify_output: str) -> str:
    return f"""You are Codex. Fix the repo so that the verify commands pass.

Rules:
{rules}

Current task: {task.id} — {task.title}

The verify run failed. Here is the full output:
--- VERIFY OUTPUT START ---
{verify_output}
--- VERIFY OUTPUT END ---

Instructions:
- Fix ONLY what is needed to make the verify commands pass.
- Do NOT refactor unrelated code.
- Keep changes minimal.
Return a short summary of what you changed.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Local PR-agent loop using Codex + gh.")
    ap.add_argument("--repo", default=".", help="Path to git repo root.")
    ap.add_argument("--todo", default="ai-pr-loop/todo.yaml", help="Path to todo.yaml.")
    ap.add_argument("--dry-run", action="store_true", help="Do not run Codex or create PRs; just print plan.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    todo_path = Path(args.todo).resolve()

    if not todo_path.exists():
        print(f"todo file not found: {todo_path}")
        print("Create it by copying ai-pr-loop/todo.example.yaml to ai-pr-loop/todo.yaml and editing.")
        return 2

    project, settings, tasks = load_todo(todo_path)
    state_path = todo_path.parent / STATE_FILENAME
    state = load_state(state_path)
    completed = state.get("completed_tasks", {})

    print(f"Repo: {repo}")
    print(f"Base branch: {project.base_branch}")
    print(f"Tasks total: {len(tasks)}")
    print(f"Completed: {len(completed)}")

    if args.dry_run:
        for t in tasks:
            if t.id in completed:
                continue
            branch = f"ai/{slugify(t.id)}"
            print(f"\nNEXT TASK: {t.id} -> branch {branch}")
            print(f"  title: {t.title}")
            print(f"  verify: {t.verify}")
        return 0

    # Preflight
    ensure_clean_git(repo, settings.allow_dirty_worktree)

    rules = read_rules(repo)
    if not rules:
        print("Warning: AI_RULES.md not found or empty. Create ai-pr-loop/AI_RULES.md for better behavior.")

    for t in tasks:
        if t.id in completed:
            continue

        branch = f"ai/{slugify(t.id)}"
        pr_title = f"{t.title}"
        pr_body = f"""Automated PR for task `{t.id}`.

### Acceptance criteria
{chr(10).join([f"- [ ] {x}" for x in t.acceptance_criteria]) if t.acceptance_criteria else "- [ ] (none provided)"}

### How to verify
{chr(10).join([f"- `{x}`" for x in t.verify])}
"""

        print(f"\n=== Working on task: {t.id} ({t.title}) ===")
        print(f"Branch: {branch}")

        # Start from base
        checkout_base(repo, project.base_branch)

        # Create branch (if exists, reuse)
        code, _ = sh(["git", "rev-parse", "--verify", branch], cwd=repo, check=False)
        if code == 0:
            print(f"Branch already exists locally; checking out: {branch}")
            sh(["git", "checkout", branch], cwd=repo, check=True)
        else:
            create_branch(repo, branch)

        attempts = 0
        success = False
        last_verify_out = ""

        while attempts < settings.max_attempts_per_task and not success:
            attempts += 1
            print(f"\n-- Attempt {attempts}/{settings.max_attempts_per_task} --")

            if attempts == 1:
                prompt = build_task_prompt(project, t, rules)
            else:
                prompt = build_fix_prompt(t, rules, last_verify_out)

            # Run Codex
            try:
                codex_out = codex_exec(repo, prompt, json_output=True)
                print("\n[Codex output captured]")
                # Optional: save codex output per attempt
                (todo_path.parent / f"codex_{t.id}_attempt{attempts}.log").write_text(codex_out, encoding="utf-8")
            except Exception as e:
                print(f"Codex exec failed: {e}")
                break

            # Verify
            ok, verify_out = run_verify(repo, t.verify)
            (todo_path.parent / f"verify_{t.id}_attempt{attempts}.log").write_text(verify_out, encoding="utf-8")
            last_verify_out = verify_out

            if ok:
                print("Verify: PASS ✅")
                success = True
            else:
                print("Verify: FAIL ❌ (will retry if attempts remain)")

        if not success:
            print(f"\nTask {t.id} did not reach passing verify within attempts. Leaving branch for manual inspection.")
            completed[t.id] = {"branch": branch, "pr_url": None, "attempts": attempts, "status": "failed"}
            state["completed_tasks"] = completed
            save_state(state_path, state)
            continue

        # Commit/push/PR
        try:
            commit_all(repo, f"{t.title} ({t.id})")
            push_branch(repo, branch)
            pr_url = create_pr(
                repo=repo,
                title=pr_title,
                body=pr_body,
                base=project.base_branch,
                head=branch,
                draft=settings.pr_draft,
            )
            print(f"PR created: {pr_url}")
            completed[t.id] = {"branch": branch, "pr_url": pr_url, "attempts": attempts, "status": "done"}
        except Exception as e:
            print(f"Failed during commit/push/PR creation: {e}")
            completed[t.id] = {"branch": branch, "pr_url": None, "attempts": attempts, "status": "needs_manual"}
        finally:
            state["completed_tasks"] = completed
            save_state(state_path, state)

    print("\nAll tasks processed (or skipped). See ai-pr-loop/state.json for results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

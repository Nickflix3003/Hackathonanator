import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


STATE_FILENAME = "state.json"
LOG_DIRNAME = "logs"
CODEX_PROMPT_FILE_SUPPORTED: Optional[bool] = None
CODEX_PROMPT_FILE_WARNING_EMITTED = False


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
    stop_on_failure: bool
    max_files_changed: int
    max_lines_changed: int
    max_run_minutes: int
    max_task_minutes: int
    codex_timeout_seconds: int
    verify_timeout_seconds: int
    max_verify_output_chars: int
    max_prompt_chars: int


@dataclass
class Task:
    id: str
    title: str
    description: str
    acceptance_criteria: List[str]
    verify: List[str]


class CommandTimeoutError(RuntimeError):
    pass


def sh(
    cmd: List[str],
    cwd: Path,
    check: bool = True,
    capture: bool = True,
    timeout_seconds: Optional[float] = None,
) -> Tuple[int, str]:
    """Run a shell command and return (exit_code, combined_output)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            env=os.environ.copy(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = ""
        if getattr(exc, "stdout", None):
            output += exc.stdout
        if getattr(exc, "stderr", None):
            output += exc.stderr
        raise CommandTimeoutError(
            f"Command timed out after {timeout_seconds}s: {' '.join(cmd)}\n\n{output}"
        ) from exc
    out = proc.stdout or ""
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\n{out}")
    return proc.returncode, out


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found in PATH.")


def codex_supports_prompt_file(repo: Path) -> bool:
    global CODEX_PROMPT_FILE_SUPPORTED
    if CODEX_PROMPT_FILE_SUPPORTED is not None:
        return CODEX_PROMPT_FILE_SUPPORTED
    try:
        code, out = sh(["codex", "exec", "--help"], cwd=repo, check=False, capture=True)
    except Exception:
        CODEX_PROMPT_FILE_SUPPORTED = False
        return CODEX_PROMPT_FILE_SUPPORTED
    CODEX_PROMPT_FILE_SUPPORTED = code == 0 and "--prompt-file" in out
    return CODEX_PROMPT_FILE_SUPPORTED


def ensure_git_repo(repo: Path) -> None:
    code, _ = sh(["git", "rev-parse", "--show-toplevel"], cwd=repo, check=False)
    if code != 0:
        raise RuntimeError(f"Not a git repository: {repo}")


def coerce_str_list(value: Any, label: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    raise ValueError(f"{label} must be a list of strings.")


def coerce_int(value: Any, default: int, label: str) -> int:
    if value is None or value == "":
        return default
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if ivalue < 0:
        raise ValueError(f"{label} must be >= 0.")
    return ivalue


def truncate_tail(text: str, max_chars: int, label: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    notice = f"[{label} truncated; showing last {max_chars} chars]\n"
    if len(notice) >= max_chars:
        return text[-max_chars:]
    tail_len = max_chars - len(notice)
    return notice + text[-tail_len:]


def remaining_seconds(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def clamp_timeout(configured_seconds: int, remaining: Optional[float]) -> Optional[float]:
    timeout = configured_seconds if configured_seconds > 0 else None
    if remaining is None:
        return timeout
    if timeout is None:
        return remaining
    return min(timeout, remaining)


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:60] if len(s) > 60 else s


def load_todo(todo_path: Path) -> Tuple[ProjectConfig, SettingsConfig, List[Task]]:
    data = yaml.safe_load(todo_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("todo.yaml must be a mapping with keys: project, settings, tasks. See todo.example.yaml.")

    project = data.get("project", {})
    settings = data.get("settings", {})
    tasks_raw = data.get("tasks", [])

    if project and not isinstance(project, dict):
        raise ValueError("project must be a mapping.")
    if settings and not isinstance(settings, dict):
        raise ValueError("settings must be a mapping.")
    if tasks_raw is None:
        tasks_raw = []
    if not isinstance(tasks_raw, list):
        raise ValueError("tasks must be a list of task mappings.")

    proj = ProjectConfig(
        name=str(project.get("name", "Project")).strip(),
        base_branch=str(project.get("base_branch", "main")).strip(),
        default_verify=coerce_str_list(project.get("default_verify", []), "project.default_verify"),
    )
    if not proj.base_branch:
        raise ValueError("project.base_branch must be a non-empty string.")

    sett = SettingsConfig(
        max_attempts_per_task=coerce_int(settings.get("max_attempts_per_task"), 3, "settings.max_attempts_per_task"),
        pr_draft=bool(settings.get("pr_draft", True)),
        allow_dirty_worktree=bool(settings.get("allow_dirty_worktree", False)),
        stop_on_failure=bool(settings.get("stop_on_failure", True)),
        max_files_changed=coerce_int(settings.get("max_files_changed"), 0, "settings.max_files_changed"),
        max_lines_changed=coerce_int(settings.get("max_lines_changed"), 0, "settings.max_lines_changed"),
        max_run_minutes=coerce_int(settings.get("max_run_minutes"), 0, "settings.max_run_minutes"),
        max_task_minutes=coerce_int(settings.get("max_task_minutes"), 15, "settings.max_task_minutes"),
        codex_timeout_seconds=coerce_int(settings.get("codex_timeout_seconds"), 600, "settings.codex_timeout_seconds"),
        verify_timeout_seconds=coerce_int(settings.get("verify_timeout_seconds"), 600, "settings.verify_timeout_seconds"),
        max_verify_output_chars=coerce_int(
            settings.get("max_verify_output_chars"), 12000, "settings.max_verify_output_chars"
        ),
        max_prompt_chars=coerce_int(settings.get("max_prompt_chars"), 30000, "settings.max_prompt_chars"),
    )

    tasks: List[Task] = []
    seen_ids = set()
    for idx, t in enumerate(tasks_raw, start=1):
        if not isinstance(t, dict):
            raise ValueError(f"Task #{idx} must be a mapping.")
        tid = str(t.get("id", "")).strip()
        if not tid:
            raise ValueError(f"Task #{idx} is missing a non-empty id.")
        if tid in seen_ids:
            raise ValueError(f"Duplicate task id: {tid}")
        seen_ids.add(tid)
        title = str(t.get("title", "")).strip()
        if not title:
            raise ValueError(f"Task '{tid}' is missing a non-empty title.")
        desc = str(t.get("description", "") or "").rstrip()
        ac = coerce_str_list(t.get("acceptance_criteria"), f"tasks[{tid}].acceptance_criteria")
        verify = coerce_str_list(t.get("verify"), f"tasks[{tid}].verify")
        if not verify:
            verify = proj.default_verify
        if not verify:
            raise ValueError(f"Task '{tid}' has no verify commands and project.default_verify is empty.")
        tasks.append(Task(id=tid, title=title, description=desc, acceptance_criteria=ac, verify=verify))

    return proj, sett, tasks


def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {state_path}: {exc}") from exc
    return {
        "completed_tasks": {},  # task_id -> {branch, pr_url, attempts}
    }


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    tmp_path = state_path.with_name(state_path.name + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp_path, state_path)


def ensure_clean_git(repo: Path, allow_dirty: bool) -> None:
    _, out = sh(["git", "status", "--porcelain"], cwd=repo, check=True)
    if out.strip() and not allow_dirty:
        raise RuntimeError(
            "Working tree is dirty. Commit/stash changes first, or set settings.allow_dirty_worktree=true."
        )


def current_branch(repo: Path) -> str:
    _, out = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, check=True)
    return out.strip()


def ensure_base_branch(repo: Path, base_branch: str) -> None:
    code, _ = sh(["git", "rev-parse", "--verify", base_branch], cwd=repo, check=False)
    if code != 0:
        raise RuntimeError(f"Base branch '{base_branch}' not found locally. Fetch or create it first.")


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


def compute_diff_stats(repo: Path, base_branch: Optional[str] = None) -> Tuple[int, int, int]:
    cmd = ["git", "diff", "--numstat"]
    if base_branch:
        cmd.append(base_branch)
    _, out = sh(cmd, cwd=repo, check=True)
    files_changed = 0
    added = 0
    deleted = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        files_changed += 1
        add_str, del_str = parts[0], parts[1]
        if add_str.isdigit():
            added += int(add_str)
        if del_str.isdigit():
            deleted += int(del_str)
    return files_changed, added, deleted


def diff_is_oversize(
    files_changed: int,
    total_lines: int,
    max_files_changed: int,
    max_lines_changed: int,
) -> Optional[str]:
    if max_files_changed > 0 and files_changed > max_files_changed:
        return f"files changed {files_changed} > limit {max_files_changed}"
    if max_lines_changed > 0 and total_lines > max_lines_changed:
        return f"lines changed {total_lines} > limit {max_lines_changed}"
    return None


def stash_changes(repo: Path, message: str) -> Optional[str]:
    _, status = sh(["git", "status", "--porcelain"], cwd=repo, check=True)
    if not status.strip():
        return None
    sh(["git", "stash", "push", "-u", "-m", message], cwd=repo, check=True)
    _, ref = sh(["git", "stash", "list", "-n", "1", "--pretty=%gd"], cwd=repo, check=True)
    return ref.strip() or None


def run_verify(
    repo: Path,
    commands: List[str],
    per_command_timeout_seconds: int,
    deadline: Optional[float],
) -> Tuple[bool, str, bool]:
    combined = []
    for c in commands:
        remaining = remaining_seconds(deadline)
        if remaining is not None and remaining <= 0:
            combined.append("Task time budget exceeded before running verify command.\n")
            return False, "\n".join(combined), True
        timeout = clamp_timeout(per_command_timeout_seconds, remaining)
        if timeout is not None and timeout <= 0:
            combined.append("Verify timeout reached before running command.\n")
            return False, "\n".join(combined), True

        combined.append(f"$ {c}\n")
        # Run via shell to support "npm test && npm run build" etc.
        try:
            proc = subprocess.run(
                c,
                cwd=str(repo),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            combined.append(output)
            combined.append(f"\nVerify command timed out after {timeout}s.\n")
            return False, "\n".join(combined), True
        combined.append(proc.stdout or "")
        if proc.returncode != 0:
            return False, "\n".join(combined), False
    return True, "\n".join(combined), False


def read_rules(repo: Path) -> str:
    rules_path = repo / "ai-pr-loop" / "AI_RULES.md"
    if rules_path.exists():
        return rules_path.read_text(encoding="utf-8").strip()
    return ""


def codex_exec(
    repo: Path,
    prompt: str,
    json_output: bool = True,
    timeout_seconds: Optional[float] = None,
    prompt_path: Optional[Path] = None,
) -> str:
    """
    Calls Codex CLI. Assumes 'codex' is installed and authenticated.
    Uses 'codex exec' so this script can run non-interactively.
    """
    global CODEX_PROMPT_FILE_WARNING_EMITTED
    cmd = ["codex", "exec"]
    if json_output:
        cmd.append("--json")
    if prompt_path and codex_supports_prompt_file(repo):
        cmd.extend(["--prompt-file", str(prompt_path)])
    else:
        if prompt_path and not CODEX_PROMPT_FILE_WARNING_EMITTED:
            print("Warning: Codex CLI does not support --prompt-file; using inline prompt.")
            CODEX_PROMPT_FILE_WARNING_EMITTED = True
        cmd.append(prompt)

    # We capture output; Codex will print logs + maybe JSON.
    _, out = sh(cmd, cwd=repo, check=True, timeout_seconds=timeout_seconds)
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


def build_fix_prompt(
    task: Task,
    rules: str,
    verify_output: str,
    max_verify_output_chars: int,
    max_prompt_chars: int,
) -> str:
    def render(output: str) -> str:
        return f"""You are Codex. Fix the repo so that the verify commands pass.

Rules:
{rules}

Current task: {task.id} - {task.title}

The verify run failed. Here is the full output:
--- VERIFY OUTPUT START ---
{output}
--- VERIFY OUTPUT END ---

Instructions:
- Fix ONLY what is needed to make the verify commands pass.
- Do NOT refactor unrelated code.
- Keep changes minimal.
Return a short summary of what you changed.
"""

    clipped_output = truncate_tail(verify_output, max_verify_output_chars, "verify output")
    prompt = render(clipped_output)
    if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
        trim_to = max_prompt_chars - (len(prompt) - len(clipped_output))
        if trim_to <= 0:
            clipped_output = "[verify output omitted to fit prompt size]\n"
        else:
            clipped_output = truncate_tail(verify_output, trim_to, "verify output")
        prompt = render(clipped_output)
        if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]
    return prompt


def main() -> int:
    ap = argparse.ArgumentParser(description="Local PR-agent loop using Codex + gh.")
    ap.add_argument("--repo", default=".", help="Path to git repo root.")
    ap.add_argument("--todo", default="ai-pr-loop/todo.yaml", help="Path to todo.yaml.")
    ap.add_argument("--dry-run", action="store_true", help="Do not run Codex or create PRs; just print plan.")
    ap.add_argument(
        "--max-prs",
        type=int,
        default=0,
        help="Stop after creating this many PRs (0 means no limit).",
    )
    ap.add_argument("--rerun-failed", action="store_true", help="Re-run tasks that are not marked done.")
    ap.add_argument("--reset-branch", action="store_true", help="Reset existing task branches to base.")
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
    ensure_tool("git")
    ensure_tool("codex")
    ensure_tool("gh")
    ensure_git_repo(repo)
    ensure_base_branch(repo, project.base_branch)
    ensure_clean_git(repo, settings.allow_dirty_worktree)

    rules = read_rules(repo)
    if not rules:
        print("Warning: AI_RULES.md not found or empty. Create ai-pr-loop/AI_RULES.md for better behavior.")

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_dir = todo_path.parent / LOG_DIRNAME / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    prs_created = 0
    run_deadline = None
    if settings.max_run_minutes > 0:
        run_deadline = time.monotonic() + (settings.max_run_minutes * 60)

    for task_index, t in enumerate(tasks, start=1):
        if run_deadline is not None and time.monotonic() >= run_deadline:
            print("Run time budget exceeded. Stopping before starting next task.")
            break

        completed_entry = completed.get(t.id)
        if completed_entry and (not args.rerun_failed or completed_entry.get("status") == "done"):
            continue

        safe_id = slugify(t.id) or f"task-{task_index}"
        branch = f"ai/{safe_id}"
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
            if args.reset_branch:
                print(f"Branch already exists; resetting to {project.base_branch}: {branch}")
                sh(["git", "checkout", branch], cwd=repo, check=True)
                sh(["git", "reset", "--hard", project.base_branch], cwd=repo, check=True)
                sh(["git", "clean", "-fd"], cwd=repo, check=True)
            else:
                print(f"Branch already exists locally; checking out: {branch}")
                sh(["git", "checkout", branch], cwd=repo, check=True)
        else:
            create_branch(repo, branch)

        attempts = 0
        success = False
        last_verify_out = ""
        failure_status = "failed"
        failure_reason = None
        task_deadline = None
        if settings.max_task_minutes > 0:
            task_deadline = time.monotonic() + (settings.max_task_minutes * 60)
        if run_deadline is not None and (task_deadline is None or run_deadline < task_deadline):
            task_deadline = run_deadline

        while attempts < settings.max_attempts_per_task and not success:
            attempts += 1
            print(f"\n-- Attempt {attempts}/{settings.max_attempts_per_task} --")

            remaining = remaining_seconds(task_deadline)
            if remaining is not None and remaining <= 0:
                failure_status = "timeout"
                if run_deadline is not None and remaining_seconds(run_deadline) <= 0:
                    failure_reason = "run_time_budget_exceeded"
                else:
                    failure_reason = "task_time_budget_exceeded"
                break

            if attempts == 1:
                prompt = build_task_prompt(project, t, rules)
            else:
                prompt = build_fix_prompt(
                    t,
                    rules,
                    last_verify_out,
                    settings.max_verify_output_chars,
                    settings.max_prompt_chars,
                )
            if settings.max_prompt_chars > 0 and len(prompt) > settings.max_prompt_chars:
                prompt = prompt[:settings.max_prompt_chars]

            # Run Codex
            try:
                prompt_path = log_dir / f"prompt_{safe_id}_attempt{attempts}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                codex_timeout = clamp_timeout(settings.codex_timeout_seconds, remaining)
                if codex_timeout is not None and codex_timeout <= 0:
                    failure_status = "timeout"
                    if run_deadline is not None and remaining_seconds(run_deadline) <= 0:
                        failure_reason = "run_time_budget_exceeded"
                    else:
                        failure_reason = "task_time_budget_exceeded"
                    break
                codex_out = codex_exec(repo, prompt, json_output=True, timeout_seconds=codex_timeout, prompt_path=prompt_path)
                print("\n[Codex output captured]")
                (log_dir / f"codex_{safe_id}_attempt{attempts}.log").write_text(codex_out, encoding="utf-8")
            except CommandTimeoutError as e:
                print(f"Codex exec timed out: {e}")
                failure_status = "timeout"
                failure_reason = "codex_timeout"
                break
            except Exception as e:
                print(f"Codex exec failed: {e}")
                failure_status = "failed"
                failure_reason = "codex_exec_failed"
                break

            # Verify
            ok, verify_out, verify_timed_out = run_verify(
                repo,
                t.verify,
                settings.verify_timeout_seconds,
                task_deadline,
            )
            (log_dir / f"verify_{safe_id}_attempt{attempts}.log").write_text(verify_out, encoding="utf-8")
            last_verify_out = verify_out

            if ok:
                print("Verify: PASS.")
                success = True
            else:
                if verify_timed_out:
                    failure_status = "timeout"
                    if run_deadline is not None and remaining_seconds(run_deadline) <= 0:
                        failure_reason = "run_time_budget_exceeded"
                    elif task_deadline is not None and remaining_seconds(task_deadline) <= 0:
                        failure_reason = "task_time_budget_exceeded"
                    else:
                        failure_reason = "verify_timeout"
                    break
                print("Verify: FAIL. (will retry if attempts remain)")

        if not success:
            print(f"\nTask {t.id} did not reach passing verify within attempts. Leaving branch for manual inspection.")
            stash_ref = None
            if not settings.stop_on_failure:
                try:
                    stash_ref = stash_changes(repo, f"ai-pr-loop {t.id} failed")
                except Exception as exc:
                    print(f"Failed to stash changes: {exc}")
            completed[t.id] = {
                "branch": branch,
                "pr_url": None,
                "attempts": attempts,
                "status": failure_status,
                "reason": failure_reason,
                "stash": stash_ref,
            }
            state["completed_tasks"] = completed
            save_state(state_path, state)
            if settings.stop_on_failure:
                print("Stopping after failure to avoid dirty worktree. Fix or split the task, then rerun.")
                break
            continue

        files_changed, added, deleted = compute_diff_stats(repo, project.base_branch)
        total_lines = added + deleted
        oversize_reason = diff_is_oversize(
            files_changed,
            total_lines,
            settings.max_files_changed,
            settings.max_lines_changed,
        )
        if oversize_reason:
            print(f"\nChange size exceeds limits: {oversize_reason}. Split the task before creating a PR.")
            stash_ref = None
            if not settings.stop_on_failure:
                try:
                    stash_ref = stash_changes(repo, f"ai-pr-loop {t.id} oversize")
                except Exception as exc:
                    print(f"Failed to stash changes: {exc}")
            completed[t.id] = {
                "branch": branch,
                "pr_url": None,
                "attempts": attempts,
                "status": "too_large",
                "stash": stash_ref,
                "diff": {
                    "files_changed": files_changed,
                    "lines_changed": total_lines,
                    "added": added,
                    "deleted": deleted,
                },
            }
            state["completed_tasks"] = completed
            save_state(state_path, state)
            if settings.stop_on_failure:
                print("Stopping after oversize changes to avoid a large PR.")
                break
            continue

        # Commit/push/PR
        stop_after_prs = False
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
            prs_created += 1
            stop_after_prs = args.max_prs > 0 and prs_created >= args.max_prs
        except Exception as e:
            print(f"Failed during commit/push/PR creation: {e}")
            completed[t.id] = {"branch": branch, "pr_url": None, "attempts": attempts, "status": "needs_manual"}
            if settings.stop_on_failure:
                stop_after_prs = True
        finally:
            state["completed_tasks"] = completed
            save_state(state_path, state)
        if stop_after_prs:
            if args.max_prs > 0 and prs_created >= args.max_prs:
                print(f"Reached max PR limit ({args.max_prs}). Stopping.")
            elif settings.stop_on_failure and completed[t.id].get("status") == "needs_manual":
                print("Stopping after PR creation failure. Fix manually and rerun.")
            else:
                print("Stopping.")
            break

    print("\nAll tasks processed (or skipped). See ai-pr-loop/state.json for results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

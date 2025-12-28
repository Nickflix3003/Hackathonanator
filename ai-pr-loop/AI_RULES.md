# AI Rules (PR Agent)

## Scope / Safety
- Keep PRs small and focused on the current task only.
- Do NOT do drive-by refactors, formatting sweeps, or dependency upgrades unless the task requires it.
- Prefer minimal changes that satisfy acceptance criteria.

## Quality gates
- After implementing, ensure verify commands pass (tests/build/lint).
- If a verify command fails, fix only what’s necessary to make it pass.

## Repo hygiene
- Follow existing patterns and code style.
- Add/update tests when behavior changes (if the project has tests).
- Update documentation when user-facing behavior changes.

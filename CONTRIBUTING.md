# Contributing Guide

## Repository Etiquette

### Step 1: Sync the latest code

Before starting any work, run `git pull --rebase` to sync the latest code
from remote.

### Step 2: Create a branch

Always create a new branch before making changes. Never work directly on
`main`.

Branch structure:

- `main` — production only, never commit directly
- `dev` — integration branch, all feature branches are cut from here
- Work branches, always branched from `dev`

Work Branch naming convention: `<type>/<description>`

- `feature/` — new functionality
- `fix/` — bug fixes
- `hotfix/` — urgent production fixes
- `chore/` — maintenance, dependencies, config
- `docs/` — documentation only

Example: `feature/user-oauth`, `fix/login-timeout`

### Step 3: Make changes

Always verify you are on the correct branch (`git branch`) before editing
any files.

### Step 4: Commit & Push

Follow Conventional Commits format for all commit messages:
`<type>(<scope>): <description>`

Types must match the branch type prefix (e.g. a `fix/` branch uses `fix:`
commits).

Never push to `main`. Always push to the current feature branch.

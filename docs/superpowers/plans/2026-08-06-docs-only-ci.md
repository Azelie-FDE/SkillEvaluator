# Docs-only CI Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route `docs/**`- and `fern/**`-only pull requests through DCO, Gitleaks, and pinned Fern validation without starting the complete SkillEvaluator platform matrix.

**Architecture:** A repository-local Python classifier parses NUL-delimited Git diff records and writes a fail-closed `docs_only` job output. CI and Security workflows use job-level conditions, preserving every existing required check context while full CI remains the default for mixed changes, errors, pushes, schedules, and manual runs.

**Tech Stack:** Python 3.12, pytest, Git, GitHub Actions YAML, Fern CLI 5.66.1

---

## File structure

- Create `scripts/classify_ci_changes.py`: path-policy and Git-diff command-line tool shared by CI and Security.
- Create `tests/test_ci_change_classifier.py`: unit and temporary-Git-repository coverage for classification and fail-closed output.
- Create `tests/test_ci_workflows.py`: static contracts for triggers, job names, conditions, and always-on gates.
- Modify `.github/workflows/ci.yml`: explicit required jobs, docs validation lane, and full-lane conditions.
- Modify `.github/workflows/security.yml`: docs-only conditions for CodeQL and dependency review while keeping Gitleaks unconditional.
- Modify `CHANGELOG.md`: record the contributor-visible CI behavior.
- Modify `docs/superpowers/specs/2026-08-06-docs-only-ci-design.md`: record written-spec approval.

### Task 1: Build the fail-closed change classifier

**Files:**
- Create: `tests/test_ci_change_classifier.py`
- Create: `scripts/classify_ci_changes.py`

- [ ] **Step 1: Write pure path-policy tests**

Create parameterized tests that import `is_docs_only` and assert the exact policy:

```python
@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        ([b"docs/index.mdx"], True),
        ([b"fern/docs.yml", b"docs/assets/logo.svg"], True),
        ([b"README.md"], False),
        ([b"src/skillevaluator/tier3/reference_skills/demo/SKILL.md"], False),
        ([b"tests/golden/benchmark_tier1.md"], False),
        ([b"docs/index.mdx", b"src/skillevaluator/cli.py"], False),
        ([], False),
    ],
)
def test_is_docs_only(paths: list[bytes], expected: bool) -> None:
    assert is_docs_only(paths) is expected
```

- [ ] **Step 2: Run the path-policy tests and confirm they fail**

Run: `uv run pytest -q tests/test_ci_change_classifier.py`

Expected: collection fails because `scripts.classify_ci_changes` does not exist.

- [ ] **Step 3: Implement the pure path policy and NUL parser**

Implement these public functions in `scripts/classify_ci_changes.py`:

```python
DOC_PREFIXES = (b"docs/", b"fern/")
KNOWN_STATUSES = frozenset(b"ACDMRTUXB")


def is_docs_only(paths: Sequence[bytes]) -> bool:
    return bool(paths) and all(path.startswith(DOC_PREFIXES) for path in paths)


def parse_name_status_z(payload: bytes) -> list[bytes]:
    fields = payload.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    paths: list[bytes] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not status or status[0] not in KNOWN_STATUSES:
            raise ValueError(f"unrecognized Git status record: {status!r}")
        path_count = 2 if status[:1] in {b"R", b"C"} else 1
        if index + path_count > len(fields):
            raise ValueError(f"incomplete Git status record: {status!r}")
        paths.extend(fields[index : index + path_count])
        index += path_count
    return paths
```

- [ ] **Step 4: Add parser tests for modifications, deletions, renames, copies, and malformed records**

Use byte payloads such as `b"M\0docs/index.mdx\0"`,
`b"R100\0docs/old.mdx\0src/new.py\0"`, and
`b"R100\0docs/old.mdx\0"`. Assert that both rename paths are returned and
malformed records raise `ValueError`.

- [ ] **Step 5: Implement Git execution and command-line output**

Add SHA validation with `re.fullmatch(r"[0-9a-fA-F]{40,64}", value)`, run:

```python
subprocess.run(
    [
        "git",
        "-C",
        str(repo),
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        "--find-copies",
        f"{base}...{head}",
        "--",
    ],
    check=True,
    capture_output=True,
)
```

The CLI prints `docs_only=true` or `docs_only=false`, appends the same line to
`GITHUB_OUTPUT` when it is set, and exits zero after operational errors while
printing a `docs_only=false` diagnostic to stderr. Argument-parser errors remain
nonzero because the workflow supplies all required arguments.

- [ ] **Step 6: Add temporary-repository integration tests**

Create real Git commits in `tmp_path`, then invoke `main()` for docs-only,
mixed, deleted, renamed-out-of-docs, empty-diff, and invalid-revision cases.
Assert the process output and `GITHUB_OUTPUT` contents for each case.

- [ ] **Step 7: Run focused tests and lint**

Run: `uv run pytest -q tests/test_ci_change_classifier.py`

Expected: all classifier tests pass.

Run: `uv run ruff check scripts/classify_ci_changes.py tests/test_ci_change_classifier.py`

Expected: no lint errors.

- [ ] **Step 8: Commit the classifier**

```bash
git add scripts/classify_ci_changes.py tests/test_ci_change_classifier.py
git commit -s -m "ci: classify docs-only pull requests"
```

### Task 2: Lock down workflow contracts with tests

**Files:**
- Create: `tests/test_ci_workflows.py`

- [ ] **Step 1: Write tests for required-context invariants**

Load workflow YAML with `yaml.load(..., Loader=yaml.BaseLoader)` and assert:

```python
REQUIRED_CI_NAMES = {
    "Tests (Python 3.12)",
    "Tests (Python 3.13)",
    "Package",
    "RHEL 8 security install",
    "Tier 2 (macos-latest)",
    "Tier 2 (windows-latest)",
    "Tier 3 macOS contract and progress",
    "Native Windows local mode fails closed",
}

assert {job["name"] for job in ci["jobs"].values()} >= REQUIRED_CI_NAMES
assert "paths-ignore" not in ci["on"]["pull_request"]
assert ci["jobs"]["test-python-312"]["if"] == "${{ always() }}"
```

Also assert the seven heavy jobs use `always()` and require classifier output
to differ from `true`, the 3.12 job contains both full-test and Fern conditions,
and the Fern command reads `fern/fern.config.json` before running `fern check`.

- [ ] **Step 2: Write Security and DCO invariants**

Assert Gitleaks has no job-level `if` or `needs`, CodeQL and dependency review
retain their existing eligibility expressions plus the docs-only condition, and
DCO has no path filter or job-level condition.

- [ ] **Step 3: Run workflow tests and confirm they fail**

Run: `uv run pytest -q tests/test_ci_workflows.py`

Expected: assertions fail because the workflows still use matrices and do not
have change-classification conditions.

### Task 3: Route CI without changing required contexts

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/security.yml`

- [ ] **Step 1: Add the classifier jobs**

In each workflow, add a pull-request-only Ubuntu job named `Classify changes`.
Pin checkout by commit, use `fetch-depth: 0`, pass the event base/head SHAs as
environment variables, and expose `steps.changes.outputs.docs_only` as the job
output.

- [ ] **Step 2: Split CI matrices into explicit required jobs**

Create `test-python-312`, `test-python-313`, `tier2-macos`, and
`tier2-windows` job definitions. Preserve each existing display name, runner,
setup action, dependency sync, command, and fail-fast independence.

- [ ] **Step 3: Add fail-closed full-lane conditions**

Give all eight required jobs `needs: classify-changes`. Keep Python 3.12 at:

```yaml
if: ${{ always() }}
```

Give the other seven jobs:

```yaml
if: ${{ always() && needs.classify-changes.outputs.docs_only != 'true' }}
```

Condition the Python setup, uv setup, sync, source scan, Ruff, and pytest steps
in the 3.12 job on `docs_only != 'true'`.

- [ ] **Step 4: Add pinned Fern validation to Python 3.12**

For `docs_only == 'true'`, pin `actions/setup-node` by commit, validate that
`fern/fern.config.json` contains a numeric semantic version, install exactly
`fern-api@$FERN_VERSION`, run `fern check`, and append a docs-lane explanation
to `$GITHUB_STEP_SUMMARY`.

- [ ] **Step 5: Condition nonessential Security jobs**

Keep Gitleaks independent. Add `needs: classify-changes` and `always()` to
CodeQL and Dependency review; retain their public-repository or Advanced
Security requirements and add
`needs.classify-changes.outputs.docs_only != 'true'`.

- [ ] **Step 6: Run workflow contract tests**

Run: `uv run pytest -q tests/test_ci_workflows.py`

Expected: all workflow-contract tests pass.

- [ ] **Step 7: Validate workflow syntax and action pins**

Run: `uv run python -c 'from pathlib import Path; import yaml; [yaml.safe_load(path.read_text()) for path in Path(".github/workflows").glob("*.yml")]'`

Expected: exit zero.

Inspect every `uses:` entry in the changed workflows and confirm each external
action is pinned to a full commit SHA.

- [ ] **Step 8: Commit workflow routing**

```bash
git add .github/workflows/ci.yml .github/workflows/security.yml tests/test_ci_workflows.py
git commit -s -m "ci: skip heavy jobs for docs-only pull requests"
```

### Task 4: Document, verify, review, and publish the pull request

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the Unreleased changelog entry**

Under `Changed`, record that docs-only pull requests run DCO, Gitleaks, and
Fern validation while mixed and non-docs changes retain the full matrix.

- [ ] **Step 2: Run focused and full local verification**

Run:

```bash
uv run pytest -q tests/test_ci_change_classifier.py tests/test_ci_workflows.py
uv run ruff check .
uv run pytest -q
uv build --python 3.13 --no-sources
npx --yes fern-api@5.66.1 check --warnings
git diff --check origin/main...HEAD
```

Expected: all tests and lint pass, distributions build, Fern reports zero
errors, and Git reports no whitespace errors. The unauthenticated Fern redirect
warning is acceptable locally and must be reported.

- [ ] **Step 3: Perform the required code-review pass**

Review additions, modifications, deletions, copy/rename parsing, paths with
unusual bytes, empty diffs, invalid SHAs, classifier-job failure, push/schedule
behavior, existing check-name preservation, full-lane command parity, security
eligibility, and output/summary correctness. Fix and re-run affected tests for
every finding.

- [ ] **Step 4: Commit documentation and review fixes**

```bash
git add CHANGELOG.md docs/superpowers/specs/2026-08-06-docs-only-ci-design.md docs/superpowers/plans/2026-08-06-docs-only-ci.md
git commit -s -m "docs: record docs-only CI routing"
```

- [ ] **Step 5: Rebase safely and push**

Fetch `origin/main`, confirm the isolated worktree is clean, rebase without
discarding user changes, rerun focused verification if the base moved, and push
`feat/christopherk/docs-only-ci`.

- [ ] **Step 6: Open the dedicated pull request**

Create a ready-for-review PR whose description includes the problem, narrow
allowlist, required-check compatibility, local verification results, GitHub
outage caveat if still active, and the post-open live-canary acceptance plan.

- [ ] **Step 7: Review the live pull request**

Confirm the workflow-changing PR selects the full lane, required check names
match repository rules, DCO sees every signed commit, and no review or merge
claim is made while hosted jobs remain queued. Create the stacked docs-only and
mixed canaries only if GitHub Actions is accepting new jobs; otherwise document
the exact unverified live evidence in the PR.

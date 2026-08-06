<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

---
title: "Docs-only CI routing design"
description: "Design for validating documentation pull requests without consuming the full SkillEvaluator CI matrix."
---

**Status:** Approved
**Date:** 2026-08-06

## Context

SkillEvaluator currently starts the complete Linux, macOS, and Windows CI
matrix for every pull request. That is appropriate for code, packaging,
workflow, and runtime-artifact changes, but it makes small documentation pull
requests contend for scarce hosted runners that cannot add meaningful coverage
for those changes.

Workflow-level `paths-ignore` is not safe here. GitHub branch rules require the
existing CI check contexts, and a workflow excluded by path filtering does not
create completed contexts. The pull request can then remain blocked with those
checks pending.

## Goals

- Give pull requests that change only published documentation a small,
  deterministic validation lane.
- Keep DCO and secret scanning enabled for every pull request.
- Preserve the current required check names so this change does not require a
  coordinated repository-ruleset update.
- Fail closed: uncertain, malformed, mixed, or non-pull-request inputs run the
  complete matrix.
- Prove the behavior on GitHub, including exact required-check conclusions,
  before declaring the optimization complete.

## Non-goals

- Changing CODEOWNERS or review requirements.
- Changing the repository ruleset in the initial rollout.
- Optimizing pushes to `main`, scheduled security scans, or manually dispatched
  security scans.
- Treating every Markdown file as documentation.
- Changing documentation publishing or broadening this work into dependency
  pinning for the existing publish workflow.

## Considered approaches

### 1. Workflow-level path filters

Add `paths-ignore` to the CI and Security workflow triggers. This is the
smallest YAML change, but it can leave required checks pending because GitHub
never creates their contexts. This approach is rejected.

### 2. Classify changes and condition existing jobs

Run a lightweight classifier first, preserve the existing required contexts,
and skip expensive jobs only after GitHub has created those contexts. One
existing required Ubuntu context runs documentation validation in the docs-only
lane so invalid documentation still blocks the pull request. This is the
recommended initial design because it is repository-local and needs no ruleset
transition.

### 3. Introduce one aggregate required gate

Add separate `Docs`, `CI`, and `Security` jobs with a final aggregate gate, then
change the ruleset to require only that gate. This has the clearest long-term
shape, but changing branch protection at the same time makes rollout and
rollback more complex. It remains a possible follow-up after the first design
has live evidence.

## Change classification

A small repository-local script classifies the changed paths between the pull
request's base and head commits. It has two results:

- `docs_only=true` only when at least one path changed and every changed path is
  under `docs/**` or `fern/**`.
- `docs_only=false` for every other case.

The allowlist is intentionally narrow:

- Root `README.md` remains full CI because it is package metadata through
  `pyproject.toml`.
- Markdown under `src/skillevaluator/tier3/reference_skills/**` remains full CI
  because it is packaged runtime data.
- Markdown under `tests/**` remains full CI because it contains fixtures and
  golden outputs.
- Workflow, script, dependency, configuration, and mixed docs/code changes all
  remain full CI.

The classifier accounts for additions, modifications, deletions, copies, and
both sides of renames. Missing SHAs, an empty diff, a Git error, or an
unrecognized path record returns `docs_only=false` and an actionable diagnostic
rather than silently reducing coverage. Full-lane job conditions use
`always()` and treat a missing classifier output as `false`, so even a failed
classification job cannot select the reduced lane.

Only `pull_request` events are eligible for docs-only routing. Pushes to `main`
run full CI. Scheduled and manually dispatched Security workflows also run all
applicable security jobs.

## Workflow design

### CI workflow

The CI workflow gains a lightweight Ubuntu classification job. The existing
matrix-generated required jobs for Python tests and Tier 2 are split into
explicit jobs while retaining these exact check names:

- `Tests (Python 3.12)`
- `Tests (Python 3.13)`
- `Package`
- `RHEL 8 security install`
- `Tier 2 (macos-latest)`
- `Tier 2 (windows-latest)`
- `Tier 3 macOS contract and progress`
- `Native Windows local mode fails closed`

Explicit jobs avoid relying on how GitHub expands a matrix whose entire job is
conditionally skipped.

For a docs-only pull request, `Tests (Python 3.12)` remains the blocking
required CI context but runs a named documentation-validation step instead of
Python setup and tests. It installs the Fern CLI version recorded in
`fern/fern.config.json` (currently `5.66.1`) and runs `fern check`. The other
seven required CI jobs are conditionally skipped, creating completed skipped
contexts without requesting their Linux, macOS, Windows, or container runners.

For a mixed or non-docs pull request, all eight contexts execute their current
commands on their current operating systems. A classifier failure also selects
this full lane.

This temporary reuse of `Tests (Python 3.12)` is a compatibility bridge for the
current ruleset. The step and summary must state clearly that it validated Fern
documentation. A later aggregate-gate migration can introduce a dedicated
required `Docs` context without changing this PR's safety boundary.

### Security workflow

`Gitleaks` runs unchanged on every pull request, push, schedule, and manual run.
The Security workflow uses the same repository-local classifier for pull
requests. `CodeQL` and `Dependency review` are skipped only for docs-only pull
requests; they retain their existing eligibility conditions for all other
events. The scheduled workflow always receives the full security lane.

### DCO and publishing workflows

The DCO workflow is unchanged and remains required on every pull request. The
Publish Docs workflow is unchanged; this change validates documentation but
does not alter the main-branch publishing path.

## Testing

Unit tests cover the classifier's observable behavior:

- docs-only changes under `docs/`;
- Fern-only changes under `fern/`;
- multiple docs and Fern paths;
- mixed docs and Python paths;
- root `README.md`;
- packaged reference-skill `SKILL.md` files;
- test fixtures and golden Markdown;
- workflow and configuration changes;
- deleted files;
- renamed files, checking old and new paths;
- empty changes and invalid Git revisions.

Static workflow checks verify that trigger filters do not exclude required
contexts, the eight required check names remain exact, DCO and Gitleaks are
unconditional, and non-pull-request events fail closed to the full lane.

The repository's complete local test suite must still pass after the change.

## Live verification and rollout

The implementation is developed on `feat/christopherk/docs-only-ci` in an
isolated worktree and submitted as one dedicated pull request. The pull request
itself changes workflows and scripts, so it must exercise the full CI and
Security lanes.

Before merge, create a temporary stacked docs-only canary pull request from the
implementation branch. Its live GitHub checks must demonstrate all of the
following on one SHA:

- DCO and Gitleaks run and pass.
- `Tests (Python 3.12)` runs `fern check` and passes.
- The other seven required CI contexts conclude as skipped rather than pending.
- No macOS, Windows, RHEL container, package, Python-test, CodeQL, or dependency
  runner job starts for the docs-only change.
- GitHub reports the canary as mergeable with respect to required checks.

A second canary containing a docs change plus a harmless non-docs change must
show that the entire matrix returns. If the GitHub Actions incident prevents
jobs from starting, keep the PR open and report local/static evidence
separately; queued jobs are not live proof. Do not weaken checks to work around
an outage.

The canaries are closed after evidence is captured. No repository-ruleset
change is part of this initial rollout. If any required context is absent or
pending in the docs-only canary, do not merge the implementation; use the
aggregate-gate approach with an explicitly coordinated ruleset migration.

## Failure and rollback behavior

- Classifier errors select full CI, never the reduced lane.
- Fern failures fail the existing required `Tests (Python 3.12)` context.
- A GitHub-hosted runner outage remains distinguishable from a repository or
  workflow defect; the change is not considered verified during an outage.
- Reverting the dedicated pull request restores the existing always-full
  workflows without any repository-settings rollback.

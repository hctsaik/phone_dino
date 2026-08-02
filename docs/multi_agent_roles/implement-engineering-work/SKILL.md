---
name: implement-engineering-work
description: "Implement an approved product requirement and technical design in an existing codebase. Use when Codex needs to plan and make scoped code changes, preserve repository conventions, add tests, run focused and regression validation, update affected documentation, and return an evidence-based engineering handoff for review."
---

# Implement Engineering Work

Act as the implementation engineer in a multi-agent workflow. Deliver the smallest maintainable change that satisfies the approved requirement and architecture.

## Workflow

1. Read repository instructions, requirements, design decisions, relevant code, tests, and current worktree state.
2. Form a short implementation plan and identify risky assumptions before editing.
3. Implement in small local changes while preserving unrelated user work.
4. Add or update focused tests for new behavior and regression risks.
5. Run the smallest relevant validation first, then broader gates proportional to impact.
6. Update documentation and generated artifacts affected by the change.
7. Review the final diff for scope, correctness, security, and maintainability.

## Deliverable

Return an engineering handoff containing:

- implemented behavior and changed files;
- design deviations and rationale;
- tests and validation results;
- known limitations or residual risks;
- reviewer instructions and any follow-up decisions.

## Role Boundaries

- Do not invent product behavior when requirements are materially ambiguous.
- Do not silently change approved contracts or architecture.
- Do not claim completion when required validation has not run; report the exact gap.
- When delegated, modify only the assigned scope and report conflicts to the coordinating agent.

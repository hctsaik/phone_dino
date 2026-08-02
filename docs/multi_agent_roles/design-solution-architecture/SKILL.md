---
name: design-solution-architecture
description: "Design a feasible technical solution from approved product requirements. Use when Codex needs to evaluate architecture options, define components and interfaces, model data and state, address security, reliability, performance and operability, plan migration or rollout, record tradeoffs, and create an implementation-ready handoff."
---

# Design Solution Architecture

Act as the solution architect in a multi-agent workflow. Prefer the smallest architecture that satisfies accepted requirements and makes risks testable.

## Workflow

1. Trace product requirements to technical capabilities and quality attributes.
2. Inspect the existing system, constraints, conventions, and reusable components.
3. Compare viable options and record material tradeoffs.
4. Define boundaries, responsibilities, interfaces, data ownership, and state transitions.
5. Address failure handling, security, privacy, observability, deployment, rollback, and compatibility.
6. Identify feasibility spikes and validation gates for the highest-risk assumptions.
7. Split the design into implementation increments with clear dependencies.

## Deliverable

Produce an implementation handoff containing:

- architecture summary and context;
- component and data-flow design;
- interfaces, contracts, and state model;
- key decisions and rejected alternatives;
- risks, mitigations, validation gates, and rollout plan;
- implementation sequence and ownership boundaries.

## Role Boundaries

- Do not change product scope silently; send requirement conflicts back to the product owner.
- Do not over-design speculative future needs.
- Do not claim feasibility without identifying evidence or a validation method.
- When delegated, return the decision-ready design rather than implementing it unless explicitly assigned both roles.

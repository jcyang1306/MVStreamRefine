# Cursor / Codex Task Template

Implement **only phase PXX** defined in `subplans/XX_NAME.md`.

Read and follow, in this order:
1. `MASTER_PLAN.md`
2. `contracts/DATA_CONTRACTS.md`
3. `contracts/COORDINATES_AND_TIMING.md`
4. `contracts/SESSION_FORMAT.md` and `decisions/ADR-*.md`
5. target subplan
6. relevant integration gate in `INTEGRATION_PLAN.md`

Before editing: identify impacted paths, public interfaces, dependencies, test fixtures, assumptions and risks. Do not modify other modules except explicitly allowed interface stubs or agreed contract update.

Implement phase deliverables and quality gates; do not replace missing algorithms with silently successful placeholders. Use analytic unit tests plus deterministic replay of registered real sessions; never synthetic datasets. Report commands executed, results and measurements, files changed, open risks, and stage PASS/BLOCKED. If contract conflict is found, propose ADR and stop that conflicting change rather than privately redefining types, frames or time units.

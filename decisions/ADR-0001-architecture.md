# ADR-0001: Object-centric tracking-first architecture
Status: Proposed. Date: 2026-10-27.

Decision: clean V2 project, retain V1 as reference. Tracking estimates `T_cam_obj` from RGB-D; robot FK/handeye provides `T_base_cam` prior; object model fixed in an immutable object reference; sliding-window optimizer revises poses; mapping commits validated measurements with full rebuild baseline.

Rationale: independent object motion is not derivable from robot pose; TSDF alone cannot infer 6D motion; high-precision map requires pose uncertainty and historical reintegration.

Consequences: additional keyframe storage, asynchronous pose versions, more demanding integration tests; online precision unproven pending benchmarks.

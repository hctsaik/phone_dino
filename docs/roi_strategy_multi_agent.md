# ROI strategy — multi-agent review

## Decision

The yellow Canny contour may guide framing and provide a candidate boundary, but it must not be dilated directly into the production defect ROI. The current contour is edge pixels, may be disconnected or open, and can include internal controls or background edges. Existing `dilateContourMask()` is presentation-only for this reason.

Use four explicitly separated masks in canonical target coordinates:

1. `contextROI`: a bounded contour expansion for crop and capture guidance only.
2. `bodyROI` / `inspectionROI`: an immutable, filled equipment polygon approved during Golden review. A small recipe-bound margin may be applied after canonical alignment.
3. `alignmentMask`: stable target features only; inspection and unknown-defect areas are excluded.
4. `heldOutMask`: independent residual validation features; never used to fit the transform.

The masks must be pairwise disjoint where required by the artifact, versioned, hash-bound, and bounded in `TARGET_CANONICAL` coordinates. `mask ⊆ inspectionROI`, and residual/heatmap pixels outside the ROI must be zero.

## Recommended geometry

During Golden review, an operator should close or draw the equipment polygon while the yellow contour remains a visual guide. After ChArUco plane normalization and independent target alignment, map that polygon into the canonical target image and apply a recipe-qualified margin. Do not use fixed raw-photo pixels. A provisional implementation may use a reviewed normalized rectangle, but it must be labeled engineering-only and must not silently become a full-frame fallback.

If polygon derivation is temporarily automated, require contour closure, no self-intersection, connected-component selection, area/bounds checks, border-contact limits, and cross-frame stability. Never use an unrestricted convex hull to bridge missing edges.

## Expected behavior for the observed photos

- A blue pen on the table outside `bodyROI` is excluded from device defect evidence.
- A blue object entering `bodyROI` becomes a defect candidate.
- A blue object intersecting only the edge band is handled as `REVIEW`/`RECAPTURE_REQUIRED` according to the Recipe, not silently ignored.
- Missing ChArUco or unsafe target-relative alignment remains `RECAPTURE_REQUIRED / NOT_RUN`.

## Contract and tests

The existing `phone_dino` `TargetAlignmentPolicy.inspectionRegions` is the correct canonical handoff. `phone_cv` `contourRegion` is a source-image guidance rectangle and must not be passed as canonical ROI without a verified transform. A new ROI semantic should require a new artifact schema/digest (for example 1.2) instead of silently changing schema 1.1.

Acceptance tests must cover normal captures, board-only and target-only movement, angle/perspective changes, an outside blue object, an inside blue object, border overlap, occlusion, ROI bounds/digest, and `heatmapOutsideMaskNonzero = 0`.

## Implemented handoff (2026-08-02)

- `phone_cv` derives a deterministic `roi-v1` candidate when a Golden template is created, stores its digest, and exposes a one-time owner confirmation endpoint. The live alignment view renders the candidate rectangle over the Golden ghost image before confirmation.
- Candidates with invalid geometry, empty masks, or a greater-than-92% frame area are rejected with `INSPECTION_ROI_CANDIDATE_UNSAFE`; this is a safety bound, not semantic body segmentation.
- `phone_dino` accepts the new immutable artifact schema `1.2`; readiness invokes `require_inspection_roi()` and fails closed on missing, mismatched, or alignment/held-out-overlapping ROI regions. Schema `1.1` remains readable only for compatibility and is not the new production contract.
- The remaining qualification requirement is physical ChArUco capture validation with the actual device recipe; the current simulation evidence is not a Dino patch heatmap.

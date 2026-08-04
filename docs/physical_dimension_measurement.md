# Calibration + Segmentation physical dimension runtime

Updated: 2026-08-05

## Contract

Artifact schema `1.8`, analyzer wire schema `1.5`, and PhoneCV profile schema `1.6`
add engineering-only planar physical dimensions. The analyzer reports
`analysis.physicalDimensionEvidence` for the whole Current subject and a separate
`physicalDimensions` object on every retained candidate region. Evidence is
either `AVAILABLE` with complete metric evidence or `UNAVAILABLE` with one
reason code and no millimetre values.

This evidence never represents a defect decision, PASS/FAIL result, or certified
metrology result. Its disclaimer is
`ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF`.

## Measurement methods

The whole-subject measurement remains a direct ChArUco plane measurement. Each
candidate region has two ordered methods:

1. `CHARUCO_PLANE_CANDIDATE_MASK_MIN_AREA_RECT_V1` is preferred when this
   Current capture has qualified ChArUco calibration. Candidate contour points
   are projected directly through the Current target-to-board transform.
2. `GOLDEN_BASELINE_RATIO_CANDIDATE_MASK_MIN_AREA_RECT_V1` is used only when
   Current ChArUco is absent. It scales the candidate contour in the Current
   subject's rotated long-/short-axis frame using the active Template's
   operator-confirmed Golden length and width.

The proportional method is explicitly labelled `GOLDEN_RATIO_ESTIMATE`; it is
not silently substituted after a failed or high-residual ChArUco transform.

## ChArUco coordinate chain

1. Detect the pinned ChArUco board in the current capture.
2. Estimate the camera-image to board-plane homography and its inlier P95
   reprojection error.
3. Plane-normalize using the board's pinned square length and independent X/Y
   pixels-per-mm values.
4. Independently align the target into target-canonical coordinates.
5. Retain the inverse target-canonical-to-board-plane transform.
6. Run Current MobileSAM in target-canonical coordinates.
7. Project the Current mask contour through the inverse transform, then divide
   board-plane X/Y pixels by their corresponding pixels-per-mm values.
8. Compute a rotated minimum-area rectangle plus the mask contour area in the
   board plane.

The artifact pins the exact `profileId`, dictionary, square/marker lengths and
OpenCV marker-ID ordering. Runtime readiness returns that board definition;
PhoneCV rejects a profile mismatch before accepting an analysis request. The
current deployment profile is `COMPACT_130X90_V1` (`DICT_5X5_1000`, `7 x 5`,
`10/7 mm`, marker IDs `100..116`).

Canonical target pixels must never be divided directly by the ChArUco scale:
target alignment may translate, rotate, or rescale those pixels.

## Output

When available, the evidence contains:

- `lengthMm`: rotated rectangle long side;
- `widthMm`: rotated rectangle short side;
- `areaMm2`: Current subject contour area;
- `rotatedRectAngleDegrees`;
- detected/inlier ChArUco corners, plane reprojection error, and X/Y pixels/mm;
- `uncertainty.linearMm`, `uncertainty.areaMm2`, and relative linear uncertainty;
- the exact Current subject-mask or retained candidate-mask digest used for the
  measurement.

Candidate evidence also contains its source binding. ChArUco evidence binds the
Current calibration. Golden-ratio evidence binds the active Template ID, Golden
photo digest, confirmed Golden length/width, measurement plane, and operator
confirmation source.

The conservative uncertainty model adds the calibration reprojection residual
to the pinned MobileSAM boundary uncertainty. Linear uncertainty covers both
opposite edges. Area uncertainty uses the measured perimeter times the edge
uncertainty plus a rounded-corner term. This is an engineering estimate, not a
traceable uncertainty budget.

## Fail-closed gates

No mm values are returned when any of the following applies:

- ChArUco is absent in this capture and no qualified, operator-confirmed Golden
  ratio reference was supplied;
- the board homography or its reprojection error is outside policy;
- target alignment is not qualified;
- Current MobileSAM or the paired subject gate is not qualified;
- the Current contour is too small or invalid;
- the metric transform is invalid;
- relative linear uncertainty exceeds the artifact policy.

Target-only alignment may still run DINO when ChArUco is absent. Whole-subject
ChArUco evidence remains `UNAVAILABLE`; retained candidates may use the
Golden-ratio method. A visible ruler, full-frame pixel dimensions, stale
calibration, or an unconfirmed Golden baseline is never used as an implicit
scale.

## Physical validation still required

Before changing `approvalState` from `ENGINEERING_AUTO` to `APPROVED`, lock a
real-device dataset containing traceable known-size samples across the intended
field of view, phone distance/angle, lighting, and expected object heights.
For each axis report signed error, absolute error, repeatability, and P95 error.
Set the production correction and acceptance bounds only from that locked
dataset. Raised or non-coplanar objects require a 3-D calibration method and
must not be qualified by this planar contract.

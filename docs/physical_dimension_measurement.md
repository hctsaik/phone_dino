# Calibration + Segmentation physical dimension runtime

Updated: 2026-08-06

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

The formal coplanar whole-subject measurement and every candidate region use
direct true ChArUco plane measurement. `CHARUCO_PLANE_CANDIDATE_MASK_MIN_AREA_RECT_V1`
requires this Current capture's qualified ChArUco calibration and projects the
retained candidate contour through the Current target-to-board transform.
Golden ratios are deliberately not a measurement method: an old silhouette
cannot establish the three-dimensional plane of a new capture.

For a foreground that is not coplanar with the board, the alternate
`BACKGROUND_BOARD_PNP_FRONT_OFFSET_BODY_CROSS_SECTION_V1` method solves the
board pose from distributed outer ArUco corners and projects the Current source
mask onto a board-front plane. It keeps the full silhouette for the long side,
but reports the nominal body width from repeated central cross-sections rather
than allowing one pipe, hook, or diagonal end to set the short side. It always reports a point estimate instead of
silently treating the board plane as the object plane. The accompanying range
is explicitly tagged: fixed-rig tolerance, an uncalibrated scenario envelope,
an unvalidated learned-model interval, or a held-out calibrated 95% interval.
The front offset is either a fixed rig datum, a registered conservative prior,
or a per-photo Depth Anything V2 relative-depth correction calibrated against
the visible board's PnP camera-depth field.

The current PM-ABC-001 engineering profile uses a recipe-scoped static prior of
`70 mm` with an explicit `45-95 mm` scenario envelope. This value was selected
from the two-photo replay to avoid scaling the same physical object thickness
with camera-to-board distance; it is not a surveyed rig datum and remains
`UNCALIBRATED_SCENARIO_ENVELOPE` until a physical fixture measurement replaces
it.

Golden enrollment keeps the formal ChArUco-plane baseline fail-closed. When its
subject lies outside the board support polygon, the response remains
`physicalDimensions: UNAVAILABLE`. If the request also carries the server-owned
offset-plane calibration and the same photo has enough distributed outer ArUco
corners, it may additionally return `backgroundOffsetPlaneDimensions`. That
separate evidence is a POC engineering estimate with its own depth interval; it
never upgrades or substitutes for the Golden planar baseline.

## ChArUco coordinate chain

1. Detect the pinned ChArUco board in the current capture.
2. Estimate the camera-image to board-plane homography and its inlier P95
   reprojection error.
3. Plane-normalize using the board's pinned square length and independent X/Y
   pixels-per-mm values.
4. Independently align the target into target-canonical coordinates for DINO
   comparison only.
5. Retain both the source-to-board and target-to-source transforms.
6. Map the pinned target ROI back to the decoded source image, then run the
   metric MobileSAM pass at that original resolution. Golden enrollment uses
   the same source-resolution path.
7. Clip the source mask to the expanded inverse-mapped prompt, intersect it
   with a dilated Golden support prior when that prior retains at least half
   of the Current proposal, and discard detached connected components below
   0.5% of the main component. The prior is a support/integrity check only;
   Current pixels still define the measured contour.
8. Project the cleaned source mask contour
   through the source-to-board transform, then divide
   board-plane X/Y pixels by their corresponding pixels-per-mm values.
9. Require every transformed contour point to be inside the physical board
   support polygon; never extrapolate a board homography past that support.
10. Compute a rotated minimum-area rectangle plus the mask contour area in the
   board plane.

The artifact pins the exact `profileId`, dictionary, square/marker lengths and
OpenCV marker-ID ordering. Runtime readiness returns that board definition;
PhoneCV rejects a profile mismatch before accepting an analysis request. A
physically `VALIDATED` board is required for the formal coplanar/`APPROVED`
path. The engineering background-board PnP path may use a merely `PUBLISHED`
print, but labels its nominal geometry, front-offset prior, and wide range
instead of claiming measured physical scale.

Canonical target pixels must never be divided directly by the ChArUco scale:
target alignment may translate, rotate, or rescale those pixels.

## Output

When available, the evidence contains:

- `lengthMm`: full-silhouette long side;
- `widthMm`: rotated-rectangle short side for a true ChArUco plane, or the
  sustained central body cross-section for a background-board measurement;
- `areaMm2`: Current subject contour area;
- `rotatedRectAngleDegrees`;
- detected/inlier ChArUco corners, plane reprojection error, and X/Y pixels/mm;
- selected calibration fiducial (`CHARUCO_CORNERS` only for whole-subject
  metrics; `OUTER_ARUCO_CORNERS` is identification/alignment-only);
- `uncertainty.linearMm`, `uncertainty.areaMm2`, and relative linear uncertainty;
- for background-board measurements, `lengthLower95Mm`, `lengthUpper95Mm`,
  `widthLower95Mm`, `widthUpper95Mm`, `intervalKind`, plus the
  `depthOffsetEstimate` source and identically-tagged front-offset interval; and
- the exact Current subject-mask or retained candidate-mask digest used for the
  measurement.

Candidate evidence binds its source to the Current calibration and retained
candidate-mask digest.

The conservative uncertainty model adds the calibration reprojection residual
to the pinned MobileSAM boundary uncertainty. Linear uncertainty covers both
opposite edges. Area uncertainty uses the measured perimeter times the edge
uncertainty plus a rounded-corner term. This is an engineering estimate, not a
traceable uncertainty budget. The uncalibrated scenario envelope is intentionally
broad: it describes the range induced by the registered front-offset prior, not
a statistical 95% confidence interval. A close point estimate must therefore
never be presented as proof that the true 3-D dimensions are known.

## Fail-closed gates

No mm values are returned when any of the following applies:

- true ChArUco is absent in this capture;
- the board homography or its reprojection error is outside policy;
- only outer ArUco geometry is available for a **2-D ChArUco-plane** method;
  distributed outer ArUco is valid for the background-board PnP method;
- the subject contour lies outside the physical board support polygon;
- target alignment is not qualified;
- Current MobileSAM or the paired subject gate is not qualified;
- the Current contour is too small or invalid;
- the metric transform is invalid;
- a background-board PnP pose cannot be established, the native camera model
  cannot be bound to the decoded still, or its interval geometry is invalid.

Target-only alignment may still run DINO when ChArUco is absent, but
whole-subject and retained-candidate metric evidence remain `UNAVAILABLE`.
A visible ruler, full-frame pixel dimensions, stale calibration, a background
board, or a Golden baseline is never used as an implicit scale.

## Physical validation still required

Before changing `approvalState` from `ENGINEERING_AUTO` to `APPROVED`, lock a
real-device dataset containing traceable known-size samples across the intended
field of view, phone distance/angle, lighting, and expected object heights.
For each axis report signed error, absolute error, repeatability, and P95 error.
Set the production correction and acceptance bounds only from that locked
dataset. Raised or non-coplanar objects require a 3-D calibration method and
must not be qualified by this planar contract.

Depth Anything V2 remains a relative model: it becomes useful only after the
same photo's PnP board depth field affine-calibrates its inverse-depth output.
That fit validates the board plane only, not the foreground extrapolation; it
therefore remains `MODEL_UNVALIDATED_INTERVAL` until a locked held-out physical
dataset establishes its coverage. If it is unavailable, the registered prior
is retained as an `UNCALIBRATED_SCENARIO_ENVELOPE`, not misrepresented as 95%.

For an ordinary still that has no native intrinsics, a wildcard engineering rig
can opt into `BOARD_SELF_CALIBRATED_V1`: PhoneDINO derives a square focal length
from the visible board while assuming a centred principal point and negligible
distortion. It is a practical fallback, not camera calibration; the response
identifies it as `BACKGROUND_BOARD_PNP_SELF_CALIBRATED_INTRINSICS_V1` and keeps
the uncalibrated scenario range. Its PnP residual gate is deliberately wider
than the native-intrinsics gate, and that residual directly widens the reported
dimension range.

## Current two-photo replay (engineering evidence)

The two supplied Pixel 7a captures were replayed through the live service after
the detached-component filter was added. Against the operator reference band
`182-184 mm` by `58-60 mm`, the point estimates were:

| capture | as-was UI | as-is point estimate | signed error versus 183 x 59 mm |
| --- | --- | --- | --- |
| 1 | 233.70 x 81.19 mm | 187.37 x 61.58 mm | +4.37 / +2.58 mm |
| 2 | 242.76 x 88.11 mm | 178.13 x 62.31 mm | -4.87 / +3.31 mm |

The second capture exposes the remaining limitation: one board-pose prior cannot
recover an unknown object-to-board depth or a tilted foreground plane from one
silhouette. Before filtering, detached board specks made that same capture look
like roughly `184 mm`; that was false accuracy. The static 70 mm engineering
prior reduces the two-view long-axis mean absolute error from about `10.0 mm`
to `4.6 mm` in this replay, while the width mean absolute error is about `3.0
mm`. These are replay errors, not a general confidence guarantee. The service
returns the estimate and labels its depth range instead of rejecting it, while preserving
the evidence needed to improve it with a surveyed rig datum, native camera
calibration, a validated depth model, or a true multi-view capture.

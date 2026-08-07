# Relative-depth physical qualification protocol

## Status

Depth Anything V2 Small is installed as a content-addressed, local
engineering model. Its result is `MODEL_UNVALIDATED_INTERVAL`; it is not a
calibrated depth sensor and it must not be represented as a qualified 95%
interval until the held-out physical set below is complete.

## Pinned local artifact

- Source: official `DepthAnything/Depth-Anything-V2` checkout
  `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
- Source-tree digest:
  `sha256:baf982fb50be7fab0f6579c66cd590e4bdc7e09a1e1751238c826d41793acdaa`.
- Checkpoint: `depth_anything_v2_vits.pth`, 99,218,434 bytes.
- Checkpoint digest:
  `sha256:715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`.
- License: Apache-2.0. Do not replace it with Depth Anything V2 Base, Large
  or Giant checkpoints: those official checkpoints are CC-BY-NC-4.0.

The launcher verifies both digests before loading the model. A mismatch leaves
relative depth unavailable rather than loading a changed artifact.

Before physical qualification, PhoneDINO uses a no-regression selection gate:
it selects a learned posterior only when the complete posterior range is inside
the configured board/rig scenario envelope and is strictly narrower. A broad,
high-spread or contradictory learned result falls back to the explicit prior;
it cannot make an operator see a wider range merely because the model ran.

## Board-parallel dominant-plane selection

The POC rig may explicitly enable `dominantPlaneEnabled`. This does not claim
to reconstruct an arbitrary 3-D surface from one photo. It selects the densest
board-normal depth layer from the already board-calibrated relative depth map,
so it is valid only for the named front measurement plane parallel to the
background board. The current rig requires at least 35% of the segmented
subject and uses an 8 mm consensus half-width.

The reported subject-depth spread is then computed from only that selected
layer. Board-fit residual and model systematic error remain in the interval;
the selector cannot remove them. If there is no supported layer, if the chosen
posterior conflicts with the board/rig envelope, or if it is not narrower, the
result remains the explicit board-pose prior. Do not interpret a selected
layer as a measurement of handles, side faces, tilt, or total part thickness.

## Physical set required before calibration

Use traceable, measured spacers or gauge blocks to place the subject plane at
five board-normal offsets spanning the real workflow, including both endpoints
of the current 45--95 mm envelope. Record the certificate/measurement method
and uncertainty for every spacer. At every offset capture at least ten repeats
at each of these independent conditions:

1. board centre and edge placement;
2. two supported working distances;
3. two rotations of the subject; and
4. at least two supported phone/lens/resolution combinations.

Hold out one full device/view/offset combination before choosing any systematic
error value. Do not tune the model error or interval against the held-out rows.

## Capture record

One CSV row is required per still. The raw still SHA-256 and the exact
PhoneDINO response make every result reproducible.

```text
capture_id,raw_sha256,device_model,physical_camera_id,lens_role,image_width,image_height,board_id,rig_id,view_slot,working_distance_mm,known_front_offset_mm,reference_uncertainty_mm,predicted_offset_mm,lower95_mm,upper95_mm,board_fit_p95_mm,subject_spread_p95_mm,model_repo_digest,model_weights_digest
```

Rows must report `source=RELATIVE_DEPTH_BOARD_CALIBRATED_V1` and
`intervalKind=MODEL_UNVALIDATED_INTERVAL` while collecting the set. Rows that
fall back to the static prior are evidence of a capture/model failure and must
remain in the report; they cannot be silently omitted.

## Qualification report

For each device/lens/resolution and for the held-out set, report signed bias,
MAE, P95 absolute error, interval coverage, interval width and same-condition
repeatability. The report must also name the source/weight digests above,
PhoneDINO version, board profile, PnP inliers/residual and segmentation quality.

Only a reviewed report may change the systematic error, promote an interval to
`CALIBRATED_95`, or let a calibration artifact be selected for a device. This
does not authorize production PASS/FAIL or equipment actions.

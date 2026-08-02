# Real photo comparison findings

## 2026-08-02 PM-ABC-001

The Golden image and the two phone Still images are stored bytes and their SHA-256 values were verified. The latest capture is session `a349258a-e7ea-465d-ba1d-3e38e481f3ec` with SHA-256 `18d00bd090859bf6937ff8e43cb3b1c14e4f0c13b083abcadb875cfd4aed24dd`.

The two real images do not contain a detectable `DICT_4X4_50` marker, so ChArUco plane normalization cannot be claimed. Full-image feature matching also produced only 9 RANSAC inliers, which is below a safe production alignment gate. The manual quadrilateral rectification in the comparison folder is therefore a visual engineering aid only.

The latest image contains a blue object crossing the upper device. The engineering comparison folder records this as `BLUE_FOREIGN_OBJECT` in `defect.json` and `defect_detected.jpg`. This is a candidate mask, not a production defect decision.

## Root cause

`phone_cv` sessions created with `simulation=true` use `FixtureAnalyzerClient`. It chooses a canned PASS/FAIL/REVIEW scenario from the raw SHA and does not inspect uploaded pixels. Consequently, a simulation result cannot discover the blue object or validate geometric alignment.

## Required production behavior

1. Require a visible, recipe-bound ChArUco board and return `RECAPTURE_REQUIRED / NOT_RUN` when it is absent or unsafe.
2. Apply ChArUco only for plane normalization; locate the pump independently using immutable target reference regions.
3. Exclude the inspection region from alignment descriptors, and use held-out stable regions to reject parallax or ambiguous matches.
4. Compare the aligned inspection ROI against the Golden with a recipe-bound residual/defect mask. A foreign blue object must become a review/hold evidence region, not be absorbed into the transform.
5. Keep the current simulation path explicitly fixture-only. It must never be presented as visual defect evidence.

The generated comparison assets are under `phone_cv/runtime/private-blobs/comparisons/` and are intentionally labeled engineering-only.

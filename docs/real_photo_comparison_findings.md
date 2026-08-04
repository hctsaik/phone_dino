# Real photo comparison findings

## 2026-08-02 PM-ABC-001

The Golden image and the two phone Still images are stored bytes and their SHA-256 values were verified. The latest capture is session `a349258a-e7ea-465d-ba1d-3e38e481f3ec` with SHA-256 `18d00bd090859bf6937ff8e43cb3b1c14e4f0c13b083abcadb875cfd4aed24dd`.

The two real images do not contain a detectable `DICT_4X4_50` marker, so ChArUco plane normalization cannot be claimed. Full-image feature matching also produced only 9 RANSAC inliers, which is below a safe production alignment gate. The manual quadrilateral rectification in the comparison folder is therefore a visual engineering aid only.

The latest image contains a blue object crossing the upper device. The engineering comparison folder records this as `BLUE_FOREIGN_OBJECT` in `defect.json` and `defect_detected.jpg`. This is a candidate mask, not a production defect decision.

## Root cause

`phone_cv` sessions created with `simulation=true` use `FixtureAnalyzerClient`. It chooses a canned PASS/FAIL/REVIEW scenario from the raw SHA and does not inspect uploaded pixels. Consequently, a simulation result cannot discover the blue object or validate geometric alignment.

## Required production behavior

1. Use a visible, Recipe-bound ChArUco board for plane normalization when available; if it is absent, require independently pinned target-only alignment evidence and return `RECAPTURE_REQUIRED / NOT_RUN` when that evidence is unsafe.
2. Never infer the pump position from ChArUco alone; locate it independently using immutable target reference regions.
3. Exclude the inspection region from alignment descriptors, and use held-out stable regions to reject parallax or ambiguous matches.
4. Compare the aligned inspection ROI against the Golden with a Recipe-bound subject/support gate. A foreign blue object inside or touching the subject must remain a difference candidate, not be absorbed into the transform or erased by Current-image segmentation; distant background differences should be suppressed before DINO inference.
5. Keep the current simulation path explicitly fixture-only. It must never be presented as visual defect evidence.

The generated comparison assets are under `phone_cv/runtime/private-blobs/comparisons/` and are intentionally labeled engineering-only.

The 2026-08-03 implementation compiles a MobileSAM Golden subject mask into artifact schema 1.5 and returns schema-1.1 subject-gated DINO evidence. Schema 1.5 additionally binds the ROI-only scorer input and recipe analysis profile across Phone Dino and Phone CV. MobileSAM is not run on Current captures, and neither its mask nor DINO difference evidence constitutes defect proof. See [Golden 主體分割與背景抑制設計](golden_subject_segmentation_design.md).

The current v10 HTTP E2E is request `efe4499b-3aca-4350-9a25-1680704bde3b`. It used artifact digest `sha256:a0c4b6b4dad09130c55b596094619a9246c113d559dd13420603fd503945c529` and profile digest `sha256:8e9adf26e02b477493d43f4b07e4e496af865868d4d72457c14fd23398325d86`. The first request after process startup completed in `7.076 s`, without retry, and produced analysis `60390926769ba72b7179aa8dc017dfda13c24117255ac88de07f33b4b702f069`, ROI-only distance `0.07113791304085873`, `GOLDEN-ACTIVE-V10-8C6FB84F` as nearest normal, four deterministic scorer tiles shared with spatial evidence, and six retained candidates. Target-canonical SHA, ROI contract, Current／normal scorer-input SHA values, ordered tile digests and analyzed region were persisted; all 12 evidence/result files reloaded through controlled endpoints without exposing private paths or base64. The comparison remains `LIMITED` only because subject-contour alignment is engineering-only; it is not a defect decision.

The v9 request `86fd1766-0131-4c75-8cc9-6c48e4c4372c` remains the historical pre-optimization baseline. Its first process request exceeded the 60-second client bound, while a retry after lazy cache population completed in about 25.4 seconds.

The earlier v8 request `ba4b5412-26a4-4ec1-a59c-33b32688bb9d` remains a historical geometry/verifier baseline and is no longer the active runtime pin.

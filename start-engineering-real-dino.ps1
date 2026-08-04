if ([string]::IsNullOrWhiteSpace($env:PHONE_DINO_SERVICE_TOKEN)) {
    throw "Set PHONE_DINO_SERVICE_TOKEN before starting the engineering service."
}
$env:PHONE_DINO_ENABLE_ENGINEERING_FIXTURES = "false"
$env:PHONE_DINO_ENGINEERING_REAL_MODEL = "true"
$env:PHONE_DINO_ENGINEERING_CONTOUR_ALIGNMENT = "true"
$env:PHONE_DINO_ALLOW_TARGET_ONLY_ALIGNMENT = "true"
$env:PHONE_DINO_ARTIFACT_MANIFEST = (Resolve-Path "$PSScriptRoot\runtime\engineering-real-dino\engineering-real-dino-artifact-v17.json").Path
$env:PHONE_DINO_ARTIFACT_PACKAGE_DIGEST = "sha256:7d1256e66a6be99c564b648d4b88dfc026e3215f3215862ff4b97bd12d8542ef"
$env:PHONE_DINO_MODEL_REPO = (Resolve-Path "$PSScriptRoot\runtime\models\dinov2").Path
$env:PHONE_DINO_MODEL_WEIGHTS = (Resolve-Path "$PSScriptRoot\runtime\models\dinov2_vits14_pretrain.pth").Path
$env:PHONE_DINO_MODEL_REPOSITORY_VERSION = "sha256:6f2d411cf095064c503259f7539f399ef6929059d58ca86230792ace634cd063"
$env:PHONE_DINO_DEVICE = "cpu"
$env:PHONE_DINO_SUBJECT_SEGMENTER_REPO = (Resolve-Path "$PSScriptRoot\runtime\engineering-real-dino\mobile-sam-repository").Path
$env:PHONE_DINO_SUBJECT_SEGMENTER_WEIGHTS = (Resolve-Path "$PSScriptRoot\runtime\engineering-real-dino\mobile-sam-repository\weights\mobile_sam.pt").Path
$env:PHONE_DINO_SUBJECT_SEGMENTER_DEVICE = "cpu"
$env:PHONE_DINO_ANALYSIS_TIMEOUT_SECONDS = "60"
$env:PYTHONPATH = "$PSScriptRoot\src"
Set-Location $PSScriptRoot
& py -3.11 -m uvicorn phone_dino.app:app --host 127.0.0.1 --port 8082

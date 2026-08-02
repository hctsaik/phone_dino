$env:PHONE_DINO_SERVICE_TOKEN = "phonecv-local-dino-token-20260802"
$env:PHONE_DINO_ENABLE_ENGINEERING_FIXTURES = "false"
$env:PHONE_DINO_ARTIFACT_MANIFEST = (Resolve-Path "$PSScriptRoot\runtime\artifacts\PM-ABC-001-production-artifact.json").Path
$env:PHONE_DINO_ARTIFACT_PACKAGE_DIGEST = "sha256:c10aefe2b85fd7990e5666eb96b0d190b938aa13499decbe286873a2affb6c1c"
$env:PHONE_DINO_MODEL_REPO = (Resolve-Path "$PSScriptRoot\runtime\models\dinov2").Path
$env:PHONE_DINO_MODEL_WEIGHTS = (Resolve-Path "$PSScriptRoot\runtime\models\dinov2_vits14_pretrain.pth").Path
$env:PHONE_DINO_MODEL_REPOSITORY_VERSION = "sha256:6f2d411cf095064c503259f7539f399ef6929059d58ca86230792ace634cd063"
$env:PYTHONPATH = "$PSScriptRoot\src"
Set-Location $PSScriptRoot
& py -3.11 -m uvicorn phone_dino.app:app --host 127.0.0.1 --port 8081

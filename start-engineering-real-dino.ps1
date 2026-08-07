if ([string]::IsNullOrWhiteSpace($env:PHONE_DINO_SERVICE_TOKEN)) {
    throw "Set PHONE_DINO_SERVICE_TOKEN before starting the engineering service."
}
$topologyPath = if ([string]::IsNullOrWhiteSpace($env:PHONECV_LOCAL_ENGINEERING_TOPOLOGY)) {
    Join-Path $PSScriptRoot "..\phone_cv\config\local-engineering-topology.json"
} else {
    $env:PHONECV_LOCAL_ENGINEERING_TOPOLOGY
}
if (-not (Test-Path -LiteralPath $topologyPath -PathType Leaf)) {
    throw "PhoneCV local engineering topology was not found: $topologyPath"
}
$topology = Get-Content -Raw -Encoding UTF8 -LiteralPath $topologyPath | ConvertFrom-Json
if ($topology.schemaVersion -ne "phonecv.local-engineering-topology/1.0" -or
    $topology.phoneDinoAnalysisMode -ne "ENGINEERING_REAL_DINO") {
    throw "PhoneCV local engineering topology does not select the supported real DINO mode."
}
$dinoHost = [string]$topology.bindHost
$dinoPort = [int]$topology.ports.phoneDinoReal
if ($dinoHost -ne "127.0.0.1" -or $dinoPort -lt 1 -or $dinoPort -gt 65535) {
    throw "PhoneCV local engineering topology contains an invalid PhoneDINO endpoint."
}
$env:PHONE_DINO_ENABLE_ENGINEERING_FIXTURES = "false"
$env:PHONE_DINO_ENGINEERING_REAL_MODEL = "true"
$env:PHONE_DINO_ENGINEERING_CONTOUR_ALIGNMENT = "true"
$env:PHONE_DINO_ALLOW_TARGET_ONLY_ALIGNMENT = "true"
$env:PHONE_DINO_ARTIFACT_MANIFEST = (Resolve-Path "$PSScriptRoot\runtime\engineering-real-dino\engineering-real-dino-artifact-v21.json").Path
$env:PHONE_DINO_ARTIFACT_PACKAGE_DIGEST = "sha256:287f6e72e7c477ea550162ee882c7ee4f27c5f4174600994ae95e0799ba81fd8"
$env:PHONE_DINO_MODEL_REPO = (Resolve-Path "$PSScriptRoot\runtime\models\dinov2").Path
$env:PHONE_DINO_MODEL_WEIGHTS = (Resolve-Path "$PSScriptRoot\runtime\models\dinov2_vits14_pretrain.pth").Path
$env:PHONE_DINO_MODEL_REPOSITORY_VERSION = "sha256:6f2d411cf095064c503259f7539f399ef6929059d58ca86230792ace634cd063"
$env:PHONE_DINO_DEVICE = "cpu"
$env:PHONE_DINO_SUBJECT_SEGMENTER_REPO = (Resolve-Path "$PSScriptRoot\runtime\engineering-real-dino\mobile-sam-repository").Path
$env:PHONE_DINO_SUBJECT_SEGMENTER_WEIGHTS = (Resolve-Path "$PSScriptRoot\runtime\engineering-real-dino\mobile-sam-repository\weights\mobile_sam.pt").Path
$env:PHONE_DINO_SUBJECT_SEGMENTER_DEVICE = "cpu"
# Depth Anything V2 Small is the only deployed learned-depth artifact.  Its
# Apache-2.0 source and checkpoint are pinned below; Base/Large are excluded
# because their checkpoint license is CC-BY-NC-4.0.  This remains an
# engineering-only relative-depth source until the physical qualification set
# has been evaluated and pinned.
$env:PHONE_DINO_DEPTH_ANYTHING_REPO = (Resolve-Path "$PSScriptRoot\runtime\models\depth-anything-v2").Path
$env:PHONE_DINO_DEPTH_ANYTHING_WEIGHTS = (Resolve-Path "$PSScriptRoot\runtime\models\depth_anything_v2_vits.pth").Path
$env:PHONE_DINO_DEPTH_ANYTHING_ENCODER = "vits"
$env:PHONE_DINO_DEPTH_ANYTHING_DEVICE = "cpu"
$env:PHONE_DINO_DEPTH_ANYTHING_REPOSITORY_VERSION = "sha256:baf982fb50be7fab0f6579c66cd590e4bdc7e09a1e1751238c826d41793acdaa"
$env:PHONE_DINO_DEPTH_ANYTHING_WEIGHTS_SHA256 = "sha256:715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
$env:PHONE_DINO_ANALYSIS_TIMEOUT_SECONDS = "60"
$env:PYTHONPATH = "$PSScriptRoot\src"
Set-Location $PSScriptRoot
& py -3.11 -m uvicorn phone_dino.app:app --host $dinoHost --port $dinoPort
exit $LASTEXITCODE

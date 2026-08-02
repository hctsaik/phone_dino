$env:PHONE_DINO_SERVICE_TOKEN = "phonecv-local-dino-token-20260802"
$env:PHONE_DINO_ENABLE_ENGINEERING_FIXTURES = "true"
$env:PHONE_DINO_ALLOW_UNMAPPED_FIXTURE = "true"
$env:PHONE_DINO_FIXTURE_DIR = "C:\code\claude\phone_dino\.fixtures"
Set-Location "C:\code\claude\phone_dino"
& .\.venv\Scripts\uvicorn.exe phone_dino.app:app --host 127.0.0.1 --port 8080

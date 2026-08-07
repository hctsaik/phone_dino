# PhoneDINO agent instructions

## Local engineering runtime

- The canonical local port topology is `../phone_cv/config/local-engineering-topology.json`.
- The real analyzer must be started with `start-engineering-real-dino.ps1`; it reads `ports.phoneDinoReal` from that topology. Do not hard-code or infer the real analyzer port.
- Port `8080` is reserved for deterministic fixture testing. It must never be used to report availability of the real Engineering DINO analyzer.
- Verify the complete phone path with `../phone_cv/test-engineering-services.ps1`; readiness must report `analysisMode: ENGINEERING_REAL_DINO`.
- PhoneCV API is the only browser-facing gateway to PhoneDINO. Do not expose the analyzer service or its token to the browser.

## Safety boundary

- This service emits fail-closed image observations, not PASS/FAIL, equipment release/hold, or MES/PLC actions.

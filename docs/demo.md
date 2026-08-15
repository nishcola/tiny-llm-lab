# Demo Runbook

```powershell
Invoke-WebRequest https://github.com/nishcola/tiny-llm-lab/releases/download/v0.1-demo/tiny-llm-lab-demo-run-v0.1.zip -OutFile tiny-llm-lab-demo-run-v0.1.zip
Invoke-WebRequest https://github.com/nishcola/tiny-llm-lab/releases/download/v0.1-demo/SHA256SUMS.txt -OutFile SHA256SUMS.txt
Get-FileHash tiny-llm-lab-demo-run-v0.1.zip -Algorithm SHA256
Expand-Archive tiny-llm-lab-demo-run-v0.1.zip -DestinationPath checkpoints/runs
streamlit run src/tiny_llm_lab/app/streamlit_page.py -- --run checkpoints/runs/tiny-llm-lab-demo
```

# Demo Runbook

```powershell
Invoke-WebRequest https://github.com/nishcola/tiny-llm-lab/releases/download/v0.1-demo/tiny-llm-lab-demo-run-v0.1.zip -OutFile tiny-llm-lab-demo-run-v0.1.zip
Invoke-WebRequest https://github.com/nishcola/tiny-llm-lab/releases/download/v0.1-demo/SHA256SUMS.txt -OutFile SHA256SUMS.txt
Get-FileHash tiny-llm-lab-demo-run-v0.1.zip -Algorithm SHA256
Expand-Archive tiny-llm-lab-demo-run-v0.1.zip -DestinationPath checkpoints/runs
streamlit run src/tiny_llm_lab/app/streamlit_page.py -- --run checkpoints/runs/tiny-llm-lab-demo
```

Compare the SHA-256 printed by PowerShell with the value in `SHA256SUMS.txt` before extraction. The archive contains one complete timeline run: a resumable checkpoint and four inference snapshots. It remains ignored by Git.

Suggested flow:

1. Open **Quick Tour** and enter `ROMEO:`.
2. Inspect the byte-BPE token table and top next-token distribution at temperature `0.8`.
3. Choose an attention layer and head; use `To be, or not to be,` for a second prompt.
4. Switch to **Timeline** and compare the same prompt at early and late checkpoints.
5. Use **Explorers** for one temporary head-disable or MLP-unit intervention, making no semantic claim from a single visualization.

The release checkpoint is a tiny model trained on Tiny Shakespeare. It produces Shakespeare-like fragments, not reliable prose or factual answers.

# Optional Wake Strategies (HF Spaces)

> **Default:** no scheduled wake pings. The frontend warm-up meter handles cold starts after ~48h HF idle sleep.

Health URL for this project:

```text
https://alvinalias-industrial-failure.hf.space/health
```

See retail's `deploy/hf_space/OPTIONAL_WAKE_STRATEGIES.md` for the full option list (manual curl, workflow_dispatch, scheduled ping tradeoffs, HF concurrent slot limit).

**Approx. cold wake:** 30–60 s (smallest model in the portfolio batch).

Before recruiter outreach:

```bash
curl -s https://alvinalias-industrial-failure.hf.space/health
```

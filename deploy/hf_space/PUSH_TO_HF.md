# Push this staging directory to Hugging Face Spaces

## One-time setup

1. Create a Space at https://huggingface.co/new-space
   - Name: `industrial-failure`
   - SDK: **Docker**
   - Hardware: **CPU Basic**
   - Visibility: **Public**

2. Sync serving files into the staging directory (from the **GitHub repo root**, not inside `industrial-failure/`):

```bash
cd /path/to/industrial-failure-classification   # repo root — adjust to your machine
bash deploy/hf_space/sync_to_space.sh
```

3. Clone the empty Space repo **once** (skip if you already have `industrial-failure/`):

```bash
git clone https://huggingface.co/spaces/alvinalias/industrial-failure
```

4. Copy staged deploy files into the HF clone (run from **repo root**; destination is the HF clone folder):

```bash
rsync -av deploy/hf_space/ industrial-failure/ \
  --exclude sync_to_space.sh \
  --exclude PUSH_TO_HF.md \
  --exclude OPTIONAL_WAKE_STRATEGIES.md
```

If your HF clone lives elsewhere, replace `industrial-failure/` with that path. Do **not** paste the literal string `/path/to/...` from older docs.

5. Commit and push to Hugging Face:

```bash
cd industrial-failure
git add README.md Dockerfile requirements.txt .gitignore api src models
git commit -m "Deploy industrial failure API on HF Spaces"
git push
```

Space URL:

```text
https://alvinalias-industrial-failure.hf.space
```

## Verify

```bash
curl -s https://alvinalias-industrial-failure.hf.space/health
```

Expect `model_loaded: true`.

## After HF is live

1. Add `HF_TOKEN` to GitHub repo secrets (Settings → Secrets → Actions).
2. Push to `main` or trigger **Sync Industrial HF Space** manually.
3. Test Vercel preview from branch `hf-space-api` before merging frontend to `main`.
4. Suspend Render after production cutover is verified.

See `docs_local/HF_SPACES_MIGRATION.md` for the full acceptance checklist.

## Optional wake strategies

No scheduled wake ping by default. See [`OPTIONAL_WAKE_STRATEGIES.md`](OPTIONAL_WAKE_STRATEGIES.md).

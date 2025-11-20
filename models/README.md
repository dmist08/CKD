Models are intentionally ignored by default (see project `.gitignore`).

This folder contains serialized model artifacts used for inference. These `.joblib` files are large and are excluded from the repository to avoid bloating the git history.

If you want to track them in Git, enable Git LFS:

```powershell
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes
```

Alternatively, place model artifacts in a release or external storage (S3, Drive) and reference them here.

Copy `.github/workflows/` into your repo root.

Workflows:
- CI: lint/type/test/coverage (+Codecov support)
- Container: build & push to GHCR
- Release Please: automated releases
- CodeQL: static analysis
- Trivy: fs + image scanning
- Commitlint: conventional commits PR checks
- Snyk: optional if `SNYK_TOKEN` is set

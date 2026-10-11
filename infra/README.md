# Deployment

Quant runs on **Azure Container Apps** (region `canadacentral`), in two identical
environments: **staging** and **production**. Everything is deployed by GitHub Actions;
nobody deploys from a laptop.

## Pipeline

```
push to master
  └─ CI (.github/workflows/ci.yml)
       ├─ Lint (ruff) + strict types (mypy)
       ├─ Tests, Python 3.11 and 3.12, coverage gate 90%
       └─ Docker: build → run API + app → smoke tests → pip-audit → Trivy
            └─ publish ghcr.io/jasonraza/quant:<commit sha>
                 └─ Deploy staging    (deploy.yml: Bicep + image, wait, smoke tests)
                      └─ Deploy production (same image, same steps)
```

- Pull requests run CI only; nothing is published or deployed.
- Production only receives an image that already passed its smoke tests in staging.
- CI also runs every Monday, so new vulnerabilities in the image are caught even
  when no code changed. Dependabot opens weekly update pull requests.

## What runs in each environment

`infra/main.bicep` is the single source of truth:

| Resource | Purpose |
|---|---|
| Log Analytics workspace | Container logs, 30-day retention, 1 GB/day cap |
| Container Apps environment | Shared network and logging for the two apps |
| `quant-<env>-api` | FastAPI paper-trading API (`/docs`), 0.5 vCPU / 1 GiB |
| `quant-<env>-dashboard` | Streamlit app (tutorial + Lab), 1 vCPU / 2 GiB |

Both apps run the same image, scale to zero when idle (a first visit after a pause
waits for a cold start) and to at most one replica. Market data refreshes itself
inside the app twice a day. The API's paper account is a demo: it lives in the
container and resets when the app restarts.

## Access without secrets

GitHub signs in to Azure with **OIDC**: the managed identity
`id-quant-github-deploy` trusts tokens issued for this repository's `staging` and
`production` GitHub environments only, and those environments only accept the
`master` branch. The identity is Contributor on `rg-quant-staging` and
`rg-quant-prod`, nothing else. The repository stores identifiers
(`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`), no passwords.

`infra/bootstrap.sh` created all of this once and is safe to re-run.

## Rolling back

Every image is tagged with its commit SHA. In the Actions tab, run **Deploy**
manually with the environment and an older tag, e.g.
`ghcr.io/jasonraza/quant:<previous sha>`. Re-running an older successful CI run's
deploy jobs does the same.

## Cost

Scale to zero keeps usage inside the Container Apps monthly free grant. Set
`alwaysOn=true` in `main.bicep` to remove cold starts (billed continuously).

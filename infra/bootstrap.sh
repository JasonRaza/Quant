#!/usr/bin/env bash
# One-time setup, run by a subscription Owner after `az login` and `gh auth login`.
# Idempotent: safe to re-run.
#
# Creates:
#   - resource groups rg-quant-staging and rg-quant-prod
#   - a user-assigned managed identity that GitHub Actions signs in as through OIDC
#     (federated credentials: no password or secret is stored anywhere)
#   - Contributor on the two resource groups only (nothing else in the subscription)
#   - the GitHub environments and the variables the pipeline reads
set -euo pipefail

REPO="${REPO:-JasonRaza/Quant}"
LOCATION="${LOCATION:-canadacentral}"
SHARED_RG="rg-quant-shared"
IDENTITY="id-quant-github-deploy"
declare -A ENV_RG=([staging]=rg-quant-staging [production]=rg-quant-prod)

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
echo "Subscription $SUBSCRIPTION_ID, repo $REPO, region $LOCATION"

for rg in "$SHARED_RG" "${ENV_RG[@]}"; do
  az group create --name "$rg" --location "$LOCATION" --tags project=quant --output none
done

az identity create --name "$IDENTITY" --resource-group "$SHARED_RG" --location "$LOCATION" \
  --tags project=quant --output none
CLIENT_ID=$(az identity show -n "$IDENTITY" -g "$SHARED_RG" --query clientId -o tsv)
PRINCIPAL_ID=$(az identity show -n "$IDENTITY" -g "$SHARED_RG" --query principalId -o tsv)

for env in "${!ENV_RG[@]}"; do
  rg="${ENV_RG[$env]}"

  # GitHub may only sign in as this identity from a job running in this environment.
  if ! az identity federated-credential show --name "github-$env" \
      --identity-name "$IDENTITY" -g "$SHARED_RG" --output none 2>/dev/null; then
    az identity federated-credential create --name "github-$env" \
      --identity-name "$IDENTITY" -g "$SHARED_RG" \
      --issuer https://token.actions.githubusercontent.com \
      --subject "repo:$REPO:environment:$env" \
      --audiences api://AzureADTokenExchange --output none
  fi

  rg_id=$(az group show -n "$rg" --query id -o tsv)
  if [ -z "$(az role assignment list --assignee "$PRINCIPAL_ID" --scope "$rg_id" \
      --role Contributor --query '[0].id' -o tsv)" ]; then
    # A new identity can take a few seconds to replicate.
    for attempt in 1 2 3 4 5 6; do
      az role assignment create --assignee-object-id "$PRINCIPAL_ID" \
        --assignee-principal-type ServicePrincipal --role Contributor --scope "$rg_id" \
        --output none && break
      sleep 10
    done
  fi

  # Environment that only the master branch can deploy to.
  gh api --method PUT "repos/$REPO/environments/$env" --input - --silent <<JSON
{"deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
  gh api --method POST "repos/$REPO/environments/$env/deployment-branch-policies" \
    -f name=master -f type=branch --silent 2>/dev/null || true
  gh variable set RESOURCE_GROUP --env "$env" --repo "$REPO" --body "$rg"
done

# These are identifiers, not secrets: nothing here grants access on its own.
gh variable set AZURE_CLIENT_ID --repo "$REPO" --body "$CLIENT_ID"
gh variable set AZURE_TENANT_ID --repo "$REPO" --body "$TENANT_ID"
gh variable set AZURE_SUBSCRIPTION_ID --repo "$REPO" --body "$SUBSCRIPTION_ID"

echo "Done. GitHub Actions can now deploy to ${ENV_RG[*]} as $IDENTITY."

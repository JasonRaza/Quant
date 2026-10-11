// One environment (staging or production) of Quant on Azure Container Apps:
// a Log Analytics workspace, a Container Apps environment, and two apps (API, Streamlit)
// running the same image. Deployed by .github/workflows/ci.yml on every push to master.

targetScope = 'resourceGroup'

@description('Which environment this deployment is.')
@allowed([
  'staging'
  'production'
])
param environmentName string

@description('Container image to run, e.g. ghcr.io/jasonraza/quant:<git sha>.')
param image string

param location string = resourceGroup().location

@description('Keep one replica running at all times: no cold start, but billed continuously.')
param alwaysOn bool = false

var prefix = environmentName == 'production' ? 'quant-prod' : 'quant-stg'
var tags = {
  project: 'quant'
  environment: environmentName
}

var services = [
  {
    name: 'api'
    port: 8000
    healthPath: '/health'
    cpu: '0.5'
    memory: '1Gi'
  }
  {
    name: 'dashboard'
    port: 8501
    healthPath: '/_stcore/health'
    cpu: '1.0'
    memory: '2Gi'
  }
]

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    // Hard cap so a log storm can't eat the credit.
    workspaceCapping: {
      dailyQuotaGb: 1
    }
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource apps 'Microsoft.App/containerApps@2024-03-01' = [
  for svc in services: {
    name: '${prefix}-${svc.name}'
    location: location
    tags: tags
    properties: {
      managedEnvironmentId: env.id
      configuration: {
        activeRevisionsMode: 'Single'
        ingress: {
          external: true
          targetPort: svc.port
          transport: 'auto'
          allowInsecure: false
        }
      }
      template: {
        containers: [
          {
            name: svc.name
            image: image
            args: [
              svc.name
            ]
            env: [
              {
                name: 'PORT'
                value: string(svc.port)
              }
              {
                name: 'HEALTH_PATH'
                value: svc.healthPath
              }
            ]
            resources: {
              cpu: json(svc.cpu)
              memory: svc.memory
            }
            probes: [
              {
                type: 'Startup'
                httpGet: {
                  path: svc.healthPath
                  port: svc.port
                }
                initialDelaySeconds: 5
                periodSeconds: 10
                failureThreshold: 10
              }
              {
                type: 'Liveness'
                httpGet: {
                  path: svc.healthPath
                  port: svc.port
                }
                periodSeconds: 30
                failureThreshold: 3
              }
              {
                type: 'Readiness'
                httpGet: {
                  path: svc.healthPath
                  port: svc.port
                }
                periodSeconds: 10
                failureThreshold: 3
              }
            ]
          }
        ]
        scale: {
          // Scale to zero when idle (free tier); one replica max: the API keeps its
          // paper account in SQLite and Streamlit sessions live in memory.
          minReplicas: alwaysOn ? 1 : 0
          maxReplicas: 1
        }
      }
    }
  }
]

output apiUrl string = 'https://${apps[0].properties.configuration.ingress.fqdn}'
output dashboardUrl string = 'https://${apps[1].properties.configuration.ingress.fqdn}'

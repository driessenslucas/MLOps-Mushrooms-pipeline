# Azure ML Job Pipeline

**Overview**

This repository contains Azure Machine Learning (Azure ML) pipelines for training models, deploying models, and testing in a Kubernetes environment.
Pipelines

This pipeline handles the lifecycle of a machine learning model, including creating compute resources, training the model, and deploying it to a Kubernetes cluster.
Triggers

    Manual Trigger (workflow_dispatch): Allows manual triggering with specific input parameters.
    Automatic Trigger (workflow_call): This can be triggered by other workflows.

Inputs

    create_compute: Boolean to decide whether to create compute resources.
    train_model: Boolean to decide whether to train the model.
    skip_training_pipeline: Boolean to skip the training pipeline if set to true.
    deploy_model: Boolean to deploy the model to a Kubernetes cluster.
    compute_name: Name of the compute resource to use.

Test Local Runner

This pipeline is designed for testing the deployment of the model in a local Kubernetes environment.
Triggers

    Manual Trigger (workflow_dispatch).
    Automatic Trigger on push to main branch.

Steps

    Checks out the repository.
    Sets up Kubectl.
    Creates a test namespace in Kubernetes.
    Deploys the model and its dependencies to Kubernetes.
    Executes tests and provides logs.
    Cleans up the test environment.

Azure ML Automated Pushing Pipeline

This pipeline automates the process of pushing changes to the main branch.
Trigger

    Automatic Trigger on push to main branch.

Steps

    Uses the Azure AI pipeline for various operations (as defined in azure-ai.yaml).

Prerequisites

Before using these pipelines, ensure you have the following prerequisites set up:
Azure Credentials

    Azure subscription with necessary permissions to create and manage Azure ML resources.
    Service principal with contributor access to the Azure subscription.

You will need to setup a compute cluster (or instance) in Azure ML to run the pipelines. You can do this by following the instructions here.

    az ml compute create -f create-cluster.yml -g <resource_group> -w <workspace_name>
    with:
      #create-cluster.yml
      $schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json
      name: low-pri-example
      type: amlcompute
      size: STANDARD_DS3_v2
      min_instances: 0
      max_instances: 2
      idle_time_before_scale_down: 120
      tier: low_priority

    or either look in the environment.yml file for a compute instance creation yaml file or enable create create_compute in the github workflow file

you will also need to set up the pillow/tensorflow environments, these can both be found in the environment folder.

GitHub Secrets

The following secrets need to be configured in your GitHub repository for the pipelines to authenticate and interact with Azure resources:

    AZURE_CREDENTIALS: A JSON string containing your Azure service principal details. This is used for authenticating with Azure from GitHub Actions.

    Create this with:
    az ad sp create-for-rbac --name "<NAME>" --role contributor --scopes /subscriptions/<SUBSCRIPTION_ID> --json-auth

    Example:

    json

    {
      "clientId": "<CLIENT_ID>",
      "clientSecret": "<CLIENT_SECRET>",
      "subscriptionId": "<SUBSCRIPTION_ID>",
      "tenantId": "<TENANT_ID>"
    }

    DOCKER_HUB_PASSWORD: Your Docker Hub password or access token if you're pushing images to Docker Hub.

    repo_token: A GitHub token with necessary permissions for actions such as pushing container images to GitHub Container Registry.

Configurations

Ensure the following configurations are correctly set in your Azure ML environment and GitHub workflows:

    Azure ML workspace settings.
    Compute configurations in compute.yaml.
    Necessary environment variables and input parameters in the workflow files.

Getting Started

To use these pipelines, clone this repository, and make sure you have the necessary permissions and configurations set up in your Azure ML workspace.
then you will need to go into the pipelines folder and change what you need for your specific use case.
same thing with the components folder

## How to download an AI model using the Azure ML CLI

az ml model download --name <name> --version <version> --download-path <path> --resource-group <resource-group> --workspace-name <workspace-name>

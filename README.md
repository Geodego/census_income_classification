[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![dvc 2.8](https://img.shields.io/badge/dvc-2.8-blue.svg)](https://dvc.org/doc/install)
# Census Income Classification
Classification model on Census Bureau data

## Project description

The purpose of this project is to deploy a scalable pipeline as would be done in a production environment. For that 
purpose we build an API using FastAPI and deploy it using Heroku. The API run machine learning inference, a prediction 
on the Census Income Data Set. Data and models are saved on AWS s3 and we use DVC to track them.

In the process of building this model and API we:
- check performance on slices
- write a model card
- track the models and data using DVC
- use GitHub Actions and Heroku for CI/CD

The data used for training the models and performing the analysis must be saved in 
'data/census.csv'. The data currently used come from the [Census Bureau](https://archive.ics.uci.edu/ml/datasets/census+income). 

### Create environment
Make sure to have conda installed and ready, then create a new environment using the ``environment.yml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate census_bureau
```

## code organisation

## Data and Model Versioning
We use DVC to store and track both our data and models. We use AWS s3 for storage. The steps to follow for AWS and DVC 
set up are:
- In the CLI environment install the [AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)
- To use your new S3 bucket from the AWS CLI you will need to create an IAM user with the appropriate permissions. 
The full instructions can be found [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console).

## CI/CD
### CI
- Setup GitHub Actions on the repository. We use the pre-made GitHub Actions python-package-conda.yml and adapt it to
the version of python used: 3.8. This action runs pytest and flake8 on push and requires both to pass without error.
- Add AWS credentials to the action (secrets need to be made available to the workflow by creating Repository Secret).
Connect AWS to GitHub actions:
  - Add your [AWS credentials to the Action](https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions).
  - Make your secrets available to your workflow by creating Repository Secrets: 
  [Creating encrypted secrets for a repository](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository).
- Set up [DVC in the action](https://github.com/iterative/setup-dvc) and specify a command to ```bash dvc pull```, you need to add in the action steps defined in the action YAML file

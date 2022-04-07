# census_income_classification
Classification model on Census Bureau data

## Project description

The purpose of this project is to deploy a scalable pipeline as would be done in a production environment. For that 
purpose we build an API using FastAPI and deploy it using Heroku. The API run machine learning inference, a prediction 
on the Census Income Data Set. 

In the process of building this model and API we:
- check pergormance on slices
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

## CI/CD
### CI
- Setup GitHub Actions on the repository. We use the pre-made GitHub Actions python-package-conda.yml and adapt it to
the version of python used: 3.8. This action runs pytest and flake8 on push and requires both to pass without error.
- Add AWS credentials to the action (secrets need to be made available to the workflow by creating Repository Secret)

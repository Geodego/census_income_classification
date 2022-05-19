[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Pytorch 1.4](https://img.shields.io/badge/pytorch-1.4-blue.svg)](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/)
[![dvc 2.8](https://img.shields.io/badge/dvc-2.8-blue.svg)](https://dvc.org/doc/install)
[![fastapi 0.63](https://img.shields.io/badge/fastapi-0.63-blue.svg)](https://fastapi.tiangolo.com/release-notes/#0630)
[![uvicorn 0.17.6](https://img.shields.io/badge/uvicorn-0.17.6-blue.svg)](https://pypi.org/project/uvicorn/0.17.6/)
[![HEROKU 7.60.1](https://img.shields.io/badge/heroku-7.60.1-blue.svg)](https://www.heroku.com/)

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
Make sure to have conda installed and ready, then create a new environment using the ``requirements.txt``
file provided in the root of the repository and activate it:

```bash
> pip install -r requirements.txt
```

## Code organisation

## Data and Model Versioning
We use DVC to store and track both our data and models. We use AWS s3 for storage. The steps to follow for AWS and DVC 
set up are:
- In the CLI environment we install the [AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)
- To use our new S3 bucket from the AWS CLI we will need to create an IAM user with the appropriate permissions. 
The full instructions can be found [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console).
- To save file and track it with dvc we follow steps as shown in the example below:
```bash
> dvc add model/scaler.pkl
> git add model/scaler.pkl.dvc .gitignore
> git commit -m "Initial commit of tracked scaler.pkl"
> dvc push
> git push
```
## API Creation

- We create a RESTful API using FastAPI, using type hinting and a Pydantic model to ingest the body from POST. 
This implement:
  - GET on the root giving a welcome message.
  - POST that does model inference.
- Unit tests to test the API.
- To launch the API use the following command (the `reload` flag allows to make changes to the code and have them 
instantly deployed without restarting uvicorn):
```bash
> uvicorn main:app --reload
```
- Once the API is deployed we can get its docs at the following url: [http//127.0.0.1:8000/docs](http//127.0.0.1:8000/docs)

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

### CD with Heroku
 We use Heroku to run our python application that consists in an API for machine learning inference.
- Procfile:
  - The ```Procfile``` specifies that we use a web dyno which runs the command ```uvicorn```. This instruction allows 
  our API to be launched using ```uvicorn```.
  - We use the IP ```0.0.0.0``` to tell the server to listen on every open network interface. 
  - Heroku dynamically assigns the port to the ```PORT``` variable: we set the port CLI option to PORT with default 
  value 5000. Doing so we tell uvicorn which port to use.
- Heroku app creation:
  - We create a new app:
  ```bash
  > heroku create geof-census-app --buildpack heroku/python
  ```
  - add a remote to our local repository:
  ```bash
  > heroku git:remote -a geof-census-app
  ```
  - To be sure Heroku deploys with the proper python version we need to add a `runtime.txt` file at the root 
  of the directory
  - then we can deploy from our GitHub repository:
  ```bash
  > git push heroku main
  ```
- Pulling files from DVC with Heroku: 
  - We need to give Heroku the ability to pull in data from DVC upon app start up. We will install 
    a [buildpack](https://elements.heroku.com/buildpacks/heroku/heroku-buildpack-apt) that allows the installation of 
    apt-files and then define the Aptfile that contains a path to DVC:
  - in the CLI we run:
    ```bash
    > heroku buildpacks:add --index 1 heroku-community/apt
    ```
  - Then in the root project folder we create a file called `Aptfile` that specifies the release of DVC we want 
  installed, e.g. https://github.com/iterative/dvc/releases/download/1.10.1/dvc_1.10.1_amd64.deb
  - Finally, we need to add the following code block to main.py:
  ```
  import os
  
  if "DYNO" in os.environ and os.path.isdir(".dvc"):
      os.system("dvc config core.no_scm true")
      if os.system("dvc pull") != 0:
          exit("dvc pull failed")
      os.system("rm -r .dvc .apt/usr/lib/dvc")
  ```
- Set up access to AWS on Heroku:
  ```bash
  > heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy
  ```
The get method of the app con be accessed at the following url: [https://geof-census-app.herokuapp.com/](https://geof-census-app.herokuapp.com/)


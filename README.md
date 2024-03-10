# CI/CD Pipeline using Docker for Loan Approval Classifier Model

This repository demonstrates the implementation of a Continuous Integration/Continuous Deployment (CI/CD) pipeline for deploying a simple Machine Learning (ML) model using a pre-trained model and serving predictions through an API endpoint.

## Objective

The primary objective of this project is to automate the deployment of a loan approval classifier model with the following key steps:

1. **Create a Git repository:**
   - The project is hosted on GitHub at [https://github.com/Praveen76/CI-CD-Pipeline-using-Docker-for-Loan-Approval-Classifier-Model](https://github.com/Praveen76/CI-CD-Pipeline-using-Docker-for-Loan-Approval-Classifier-Model.git).

2. **Implement the API server:**
   - Use a Python web framework such as Flask or FastAPI to build the API server. The server should expose an endpoint for serving predictions from the ML model.

3. **Set up GitHub Actions workflow:**
   - Automate model testing and deployment on each push to the repository using GitHub Actions. This ensures a streamlined CI/CD process.

4. **Build Docker container:**
   - Create a Docker container for the API server, including the necessary dependencies and the pre-trained model.

5. **Use Docker Hub for image storage:**
   - Store the Docker image on Docker Hub to facilitate easy access and deployment.

6. **Deploy on AWS EC2:**
   - Utilize an AWS EC2 instance for deploying the Docker container, making the model accessible through the specified API endpoint.

7. **Set up API Gateway:**
   - Establish an API Gateway on AWS to securely expose the API endpoint.

## Project Structure

```
CI-CD-Pipeline-using-Docker-for-Loan-Approval-Classifier-Model/
├── .github/
│   └── workflows/
│       └── (contains CI/CD workflow files)
├── requirements/
│   └── (contains project dependencies)
├── src/
│   └── (source code for the Loan Approval Classifier Model)
├── src_api/
│   └── (FASTAPI source code)
├── tests/
│   └── (contains test scripts)
├── Dockerfile
├── LoanApprovalClassifier_model-0.0.1-py3-none-any.whl
├── MANIFEST.in
├── README.md
├── mypy.ini
├── pyproject.toml
└── setup.py
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Praveen76/CI-CD-Pipeline-using-Docker-for-Loan-Approval-Classifier-Model.git
   ```

2. Navigate to the project directory:
   ```bash
   cd CI-CD-Pipeline-using-Docker-for-Loan-Approval-Classifier-Model
   ```

3. Follow the deployment steps mentioned in the respective directories (`/model` and `/api_server`).

## CI/CD Workflow

The CI/CD workflow is automated through GitHub Actions. On each push to the repository, the workflow performs model testing and, if successful, triggers the deployment process.


## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or inquiries, feel free to contact me on [Linkedin](https://www.linkedin.com/in/praveen-kumar-anwla-49169266/).

## **About Me**:
I’m a seasoned Data Scientist and founder of [TowardsMachineLearning.Org](https://towardsmachinelearning.org/). I've worked on various Machine Learning, NLP, and cutting-edge deep learning frameworks to solve numerous business problems.

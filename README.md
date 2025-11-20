# ML Ops Major Assignment – End-to-End Pipeline

This repository implements the deliverables described in the PGD ML Ops Major assignment: training a classical ML model, automating evaluation with CI/CD, wrapping the model inside a Flask-powered web app, containerizing the solution, and defining Kubernetes manifests for a three-replica deployment.

## Repository Structure

- `train.py` / `test.py`: scripts to train and evaluate the DecisionTree model on the Olivetti faces dataset.
- `mlops_major/`: shared utilities (data loading, model helpers, image preprocessing).
- `app.py`: Flask web server that loads the trained model and exposes an upload form for predictions.
- `templates/` + `static/`: HTML/CSS assets for the Flask UI.
- `.github/workflows/ci.yml`: GitHub Actions workflow that runs `train.py` and `test.py` on each push.
- `Dockerfile`: container definition for serving the Flask application via Gunicorn.
- `k8s/`: baseline Kubernetes deployment (`3` replicas) and service manifests.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python train.py --max-depth 25 --min-samples-leaf 2
```

Artifacts are saved under `artifacts/savedmodel.pth` (joblib format) along with metadata describing accuracy, hyperparameters, and dataset details.

### Test

```bash
python test.py
```

This reloads the saved artifact, evaluates on the 30% test split (stratified, seed `42`), and prints the accuracy.

## Flask Inference Service

1. Ensure `artifacts/savedmodel.pth` exists (run `train.py` first).
2. Launch the app locally:

   ```bash
   flask --app app run --host 0.0.0.0 --port 5000
   ```

3. Upload any face image (RGB or grayscale). The server converts it to the expected 64×64 grayscale format before prediction.

## Docker Workflow

```bash
docker build -t <your-dockerhub-username>/mlops-major:latest .
docker push <your-dockerhub-username>/mlops-major:latest
```

Expose the image name via the `IMAGE_NAME` build arg or update the Kubernetes manifests accordingly.

## Kubernetes Deployment (3 Replicas)

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

`deployment.yaml` sets `replicas: 3` to meet the assignment requirement and exposes port `5000`. `service.yaml` defines a `LoadBalancer` (change to `NodePort` if needed).

## Git Branching Strategy (per assignment)

- `main`: Initialization with `README.md` and `.gitignore` (no merges from feature branches).
- `dev`: Model work, CI workflow, training/testing scripts (never merged back).
- `docker_cicd`: Derived from `dev` for Docker, Flask, and Kubernetes work (never merged back).

Document Git/GitHub screenshots when implementing on your actual repo.

## CI/CD Workflow

`.github/workflows/ci.yml` performs:

1. Checkout code
2. Set up Python
3. Install dependencies
4. Run `train.py` to generate `savedmodel.pth`
5. Run `test.py` and publish accuracy in job logs

## Next Steps

- Add automated unit tests for the data utilities.
- Integrate model quantization (optional stretch goal from assignment overview).
- Extend CI to build & push Docker images plus linting.

Refer to `assignment.txt` for the original instructions extracted from the PDF.


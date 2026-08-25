.DEFAULT_GOAL := help

-include .env
export GDRIVE_CLIENT_ID GDRIVE_CLIENT_SECRET KAGGLE_API_TOKEN

KAGGLE_DATASET := alanjafari/kurmed-triage/versions/1
DATA_DIR := data/raw
DVC_GDRIVE_TOKEN_DIR ?= $(HOME)/.local/state/techchallenge
DVC_GDRIVE_TOKEN_FILE ?= $(DVC_GDRIVE_TOKEN_DIR)/gdrive-user-credentials.json

.PHONY: api api-benchmark api-build api-down airflow airflow-down airflow-password airflow-reset check docker-config download-data dvc-reauth mlflow mlflow-down observability observability-down pull-data

help:
	@printf "Available targets:\n"
	@printf "  airflow        Start local Airflow standalone\n"
	@printf "  airflow-down   Stop local Airflow standalone\n"
	@printf "  airflow-password Show the generated local Airflow password\n"
	@printf "  airflow-reset  Remove local Airflow state and containers\n"
	@printf "  api            Start the complete observability stack after a champion model is promoted\n"
	@printf "  api-benchmark  Benchmark the local API HTTP latency and log aggregates to MLflow\n"
	@printf "  api-build      Build the API image\n"
	@printf "  api-down       Stop the complete observability stack\n"
	@printf "  mlflow         Start only local MLflow Tracking for the first bootstrap\n"
	@printf "  mlflow-down    Stop the local MLflow service\n"
	@printf "  observability  Start API, MLflow, Prometheus, and Grafana\n"
	@printf "  observability-down Stop and remove the observability stack containers\n"
	@printf "  docker-config  Validate the observability Compose configuration\n"
	@printf "  check          Run all repository quality checks\n"
	@printf "  download-data  Download KurMed-Triage v1 to %s\n" "$(DATA_DIR)"
	@printf "  pull-data      Download the DVC-tracked dataset\n"
	@printf "  dvc-reauth     Remove this project's local Google OAuth token and pull data again\n"

check:
	@uv run pre-commit run --all-files

airflow:
	@AIRFLOW_UID="$$(id -u)" docker compose -f compose.airflow.yml up --build

airflow-down:
	@AIRFLOW_UID="$$(id -u)" docker compose -f compose.airflow.yml down

airflow-password:
	@AIRFLOW_UID="$$(id -u)" docker compose -f compose.airflow.yml exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated

airflow-reset:
	@AIRFLOW_UID="$$(id -u)" docker compose -f compose.airflow.yml down --volumes --remove-orphans

mlflow:
	@docker compose -f compose.mlflow.yml up --build mlflow

mlflow-down:
	@docker compose -f compose.mlflow.yml stop mlflow

api:
	@docker compose -f compose.mlflow.yml up --build

api-benchmark:
	@uv run python -m techchallenge.http_benchmark

api-build:
	@docker compose -f compose.mlflow.yml build api

api-down:
	@docker compose -f compose.mlflow.yml down

observability:
	@docker compose -f compose.mlflow.yml up --build

observability-down:
	@docker compose -f compose.mlflow.yml down

docker-config:
	@docker compose -f compose.mlflow.yml config --quiet

download-data:
	@test -n "$$KAGGLE_API_TOKEN" || (printf "%s\n" "KAGGLE_API_TOKEN must be exported before running make download-data." >&2; exit 1)
	@mkdir -p "$(DATA_DIR)"
	@uv run python -c "import kagglehub; kagglehub.dataset_download('$(KAGGLE_DATASET)', output_dir='$(DATA_DIR)')"

pull-data:
	@test -n "$$GDRIVE_CLIENT_ID" || (printf "%s\n" "GDRIVE_CLIENT_ID must be exported before running make pull-data." >&2; exit 1)
	@test -n "$$GDRIVE_CLIENT_SECRET" || (printf "%s\n" "GDRIVE_CLIENT_SECRET must be exported before running make pull-data." >&2; exit 1)
	@mkdir -p "$(DVC_GDRIVE_TOKEN_DIR)"
	@DVC_NO_ANALYTICS=true uv run dvc remote modify --local gdrive gdrive_client_id "$$GDRIVE_CLIENT_ID"
	@DVC_NO_ANALYTICS=true uv run dvc remote modify --local gdrive gdrive_client_secret "$$GDRIVE_CLIENT_SECRET"
	@DVC_NO_ANALYTICS=true uv run dvc remote modify --local gdrive gdrive_user_credentials_file "$(DVC_GDRIVE_TOKEN_FILE)"
	@DVC_NO_ANALYTICS=true uv run dvc pull

dvc-reauth:
	@rm -f "$(DVC_GDRIVE_TOKEN_FILE)"
	@$(MAKE) pull-data

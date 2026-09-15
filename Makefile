.DEFAULT_GOAL := help

-include .env
export GDRIVE_CLIENT_ID GDRIVE_CLIENT_SECRET KAGGLE_API_TOKEN

DATA_DIR := data/raw
DVC_GDRIVE_TOKEN_DIR ?= $(HOME)/.local/state/techchallenge
DVC_GDRIVE_TOKEN_FILE ?= $(DVC_GDRIVE_TOKEN_DIR)/gdrive-user-credentials.json

.PHONY: api api-benchmark api-build api-down api-load-test airflow airflow-down airflow-password airflow-reset check docker-config download-data dvc-reauth mlflow mlflow-down observability observability-down presentation-evidence presentation-report presentation-slides pull-data setup

help:
	@printf "Available targets:\n"
	@printf "  airflow        Start local Airflow standalone\n"
	@printf "  airflow-down   Stop local Airflow standalone\n"
	@printf "  airflow-password Show the generated local Airflow password\n"
	@printf "  airflow-reset  Remove local Airflow state and containers\n"
	@printf "  api            Start the complete observability stack after a champion model is promoted\n"
	@printf "  api-benchmark  Benchmark the local API HTTP latency and log aggregates to MLflow\n"
	@printf "  api-load-test  Generate synthetic concurrent traffic for Grafana and Prometheus\n"
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
	@printf "  setup          Configure credentials, sync dependencies, install hooks, and fetch source data\n"
	@printf "  presentation-report Print the latest safe aggregate evidence from MLflow\n"
	@printf "  presentation-slides Generate a local slide deck filled with the latest MLflow metrics\n"
	@printf "  presentation-evidence Validate, refresh API metrics, generate traffic, and print the report\n"

check:
	@uv run pre-commit run --all-files

setup:
	@bash scripts/setup.sh

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

api-load-test:
	@uv run python -m techchallenge.api_load_test

presentation-report:
	@uv run python -m techchallenge.presentation_report

presentation-slides:
	@temporary_file="$$(mktemp .presentation-slides.XXXXXX)"; \
		trap 'rm -f "$$temporary_file"' EXIT; \
		uv run python -m techchallenge.presentation_report --slides > "$$temporary_file"; \
		mv "$$temporary_file" presentation-slides.generated.md
	@printf "%s\n" "Generated presentation-slides.generated.md"
	@printf "%s\n" "Preview: npx @marp-team/marp-cli@latest --preview presentation-slides.generated.md"

presentation-evidence: check docker-config
	@docker compose -f compose.mlflow.yml up --build --detach --wait --wait-timeout 300
	@$(MAKE) api-benchmark
	@$(MAKE) api-load-test
	@uv run python -m techchallenge.presentation_report --strict
	@$(MAKE) presentation-slides

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
	@uv run python scripts/download_kaggle_data.py

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

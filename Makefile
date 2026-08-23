.DEFAULT_GOAL := help

KAGGLE_DATASET := alanjafari/kurmed-triage/versions/1
DATA_DIR := data/raw

.PHONY: help airflow airflow-down airflow-password airflow-reset check download-data mlflow mlflow-down pull-data

help:
	@printf "Available targets:\n"
	@printf "  airflow        Start local Airflow standalone\n"
	@printf "  airflow-down   Stop local Airflow standalone\n"
	@printf "  airflow-password Show the generated local Airflow password\n"
	@printf "  airflow-reset  Remove local Airflow state and containers\n"
	@printf "  mlflow         Start local MLflow Tracking\n"
	@printf "  mlflow-down    Stop local MLflow Tracking\n"
	@printf "  check          Run all repository quality checks\n"
	@printf "  download-data  Download KurMed-Triage v1 to %s\n" "$(DATA_DIR)"
	@printf "  pull-data      Download the DVC-tracked dataset\n"

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
	@docker compose -f compose.mlflow.yml up --build

mlflow-down:
	@docker compose -f compose.mlflow.yml down

download-data:
	@test -n "$$KAGGLE_API_TOKEN" || (printf "%s\n" "KAGGLE_API_TOKEN must be exported before running make download-data." >&2; exit 1)
	@mkdir -p "$(DATA_DIR)"
	@uv run python -c "import kagglehub; kagglehub.dataset_download('$(KAGGLE_DATASET)', output_dir='$(DATA_DIR)')"

pull-data:
	@test -n "$$GDRIVE_CLIENT_ID" || (printf "%s\n" "GDRIVE_CLIENT_ID must be exported before running make pull-data." >&2; exit 1)
	@test -n "$$GDRIVE_CLIENT_SECRET" || (printf "%s\n" "GDRIVE_CLIENT_SECRET must be exported before running make pull-data." >&2; exit 1)
	@DVC_NO_ANALYTICS=true uv run dvc remote modify --local gdrive gdrive_client_id "$$GDRIVE_CLIENT_ID"
	@DVC_NO_ANALYTICS=true uv run dvc remote modify --local gdrive gdrive_client_secret "$$GDRIVE_CLIENT_SECRET"
	@DVC_NO_ANALYTICS=true uv run dvc pull

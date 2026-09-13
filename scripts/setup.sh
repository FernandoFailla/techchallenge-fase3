#!/usr/bin/env bash

set -euo pipefail

read_env_value() {
    local key="$1"
    awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' .env
}

update_env_value() {
    local key="$1"
    local value="$2"
    local temporary_file
    temporary_file="$(mktemp .env.tmp.XXXXXX)"
    awk -v key="$key" -v value="$value" '
        index($0, key "=") == 1 { print key "=" value; found=1; next }
        { print }
        END { if (!found) print key "=" value }
    ' .env > "$temporary_file"
    mv "$temporary_file" .env
}

read_secret() {
    local prompt="$1"
    local value
    IFS= read -r -s -p "$prompt" value
    printf "\n" >&2
    printf "%s" "$value"
}

if [[ ! -f .env ]]; then
    cp .env.example .env
    printf "%s\n" "Created .env from .env.example."
else
    printf "%s\n" "Using existing .env; values are preserved unless replaced interactively."
fi

uv sync --all-groups
uv run pre-commit install

google_client_id="$(read_env_value GDRIVE_CLIENT_ID)"
google_client_secret="$(read_env_value GDRIVE_CLIENT_SECRET)"
kaggle_token="$(read_env_value KAGGLE_API_TOKEN)"

if [[ -z "$google_client_id" || -z "$google_client_secret" ]] && [[ -t 0 ]]; then
    printf "%s\n" "Google Drive credentials are optional. Leave the client ID empty to use Kaggle."
    read -r -p "GDRIVE_CLIENT_ID: " google_client_id
    if [[ -n "$google_client_id" ]]; then
        google_client_secret="$(read_secret "GDRIVE_CLIENT_SECRET: ")"
    fi
fi

if [[ -n "$google_client_id" && -n "$google_client_secret" ]]; then
    update_env_value "GDRIVE_CLIENT_ID" "$google_client_id"
    update_env_value "GDRIVE_CLIENT_SECRET" "$google_client_secret"
    printf "%s\n" "Retrieving the approved versioned data with DVC."
    make pull-data
    exit 0
fi

if [[ -z "$kaggle_token" ]] && [[ -t 0 ]]; then
    kaggle_token="$(read_secret "KAGGLE_API_TOKEN: ")"
fi

if [[ -z "$kaggle_token" ]]; then
    printf "%s\n" "Google Drive credentials or KAGGLE_API_TOKEN are required to fetch data." >&2
    exit 1
fi

update_env_value "KAGGLE_API_TOKEN" "$kaggle_token"
printf "%s\n" "Retrieving the public source dataset from Kaggle."
make download-data
printf "%s\n" "Kaggle provides only the raw source. Start Airflow and run prepare_modeling_base before training."

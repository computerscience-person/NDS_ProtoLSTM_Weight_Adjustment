# Neuro-Dynamic Fighter LSTM HTTP Server

This repository hosts the lstm weight adjustment of neuro-dynamic fighter.

## Dependencies

- uv [install](https://docs.astral.sh/uv/getting-started/installation/#winget)
- pdm [install](https://pdm-project.org/latest/#recommended-installation-method)

These tools will automatically create and install python virtual environments, tools and libraries.

## Sync libraries (web server only)
`uv sync --locked`

## Sync libraries (all, including marimo notebooks)
`uv sync --locked --all-extras`

## Running the server (development)
`uv run fastapi dev app/main.py`

## Running the server (production)
`uv run fastapi run app/main.py`

## Run tests
`pdm run pytest`

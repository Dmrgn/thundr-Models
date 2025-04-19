 # Developer Guide

 This document outlines the recommended directory structure, development setup, and contribution guidelines for this project.

 ## Repository Layout

 ```text
 .
 ├── README.md                     # Project overview and quickstart
 ├── pyproject.toml                # Python project configuration
 ├── package.json                  # Node.js / TypeScript project configuration
 ├── tsconfig.json                 # TypeScript compiler settings
 ├── bun.lock                      # Bun lockfile
 ├── uv.lock                       # UV lockfile
 ├── .gitignore                    # Git ignore rules
 ├── models/                       # Produced textzap and imagezap keras models
 ├── docs/
 │   └── DEVELOPER_GUIDE.md        # This document
 ├── data/
 ├── src/
 │   └── ts/                       # TypeScript server for image sorter using bun
 ├── notebooks/                    # Jupyter notebooks for exploration
 │   ├── image_prep.ipynb
 │   ├── imagezap.ipynb
 │   ├── text_prep.ipynb
 │   └── textzap.ipynb
 ├── public/                       # Static web assets and frontend files for image sorter using bun
 ├── tests/                        # Unit and integration tests
 │   └── ts/
 └── ...
 ```

 ## Directory Details

 - **data/raw/**: Store original datasets, large assets, and static inputs. Use Git LFS or external download scripts to manage large files.
 - **data/processed/**: Outputs from preprocessing (not version controlled; regenerate as needed).
 - **src/python/**: Core Python package code. Include `__init__.py` and organize modules logically (e.g., `src/python/data`, `src/python/models`).
 - **src/ts/**: TypeScript application code. Entrypoints, utilities, and shared modules.
 - **src/jl/**: Julia scripts for dataset creation or analysis.
 - **notebooks/**: Jupyter notebooks for prototyping and data exploration. Keep version control minimal (clear outputs before committing).
 - **scripts/**: Standalone CLI scripts or wrappers for common tasks (e.g., data ingestion, batch processing).
 - **public/**: Static assets served by the frontend (images, icons, HTML templates).
 - **tests/**: Automated tests. Mirror language folders under `tests/` (e.g., `tests/python`, `tests/ts`).

 ## Development Setup

 ### Prerequisites
 - Python 3.11 (managed via `pyproject.toml` and uv)
 - Bun 1.2+
 - Git LFS (for large binary assets)

 ### Install Dependencies
 ```bash
 # Python dependencies
 uv sync

 # TypeScript dependencies (using bun)
 bun install
 ```

 ### Running Notebooks
 Launch Jupyter Lab in the repository root:
 ```bash
 jupyter lab notebooks/
 ```

 ---
# CodeBase Documentation Generator

A fast, concurrent documentation generator for Python projects that supports .docsignore patterns and maintains versioned backups.

a python project for creating a comperhansive and detailed Documentation and Explaination of the codes and purpose of them based on the input directories

## Features

- Asynchronous and concurrent documentation generation
- Support for .docsignore patterns (similar to .gitignore)
- Automatic backup and versioning
- Project-specific documentation organization
- Rich progress tracking and status reporting
- Error recovery and state persistence
- Multiple input methods (direct targets or file-based)

## Installation

```bash
pip install codebase-doc
```

## Usage

Generate documentation for specific directories:
```bash
docgen --target /path/to/dir1 --target /path/to/dir2
```

Use a file containing target directories:
```bash
docgen --file targets.txt
```

Combine both methods:
```bash
docgen --target /path/to/dir1 --file targets.txt
```

Force overwrite existing documentation:
```bash
docgen --target /path/to/dir1 --force
```

Specify custom output directory:
```bash
docgen --target /path/to/dir1 --output-dir /custom/path
```

## .docsignore Format

Create a `.docsignore` file in your project directory to specify ignore patterns:

```
# Ignore Python cache
__pycache__
*.pyc

# Ignore version control
.git
.svn

# Ignore virtual environments
venv
.venv

# Ignore IDE files
.idea
.vscode

# Ignore build artifacts
build
dist
*.egg-info
```

## Output Structure

```
/root/docs/
└── project_name/
    ├── latest/
    │   └── (current documentation)
    └── backup/
        └── docs_backup_YYYYMMDD_HHMMSS/
```

## Development

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest tests/`

## Contributing

Contributions are welcome! Please check our contributing guidelines for more details.

## License

This project is licensed under the MIT License.
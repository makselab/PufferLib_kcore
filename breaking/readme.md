# Temporary Folder for Breaking Symmetries

This folder contains temporary files needed to run **breaking symmetries**.  

## Current Usage (Hard Method)
To run the breaking symmetries functionality, you need to **replace** the existing files in the repository with the versions provided in this folder:
- `demo.py` → substitute the existing `demo.py`
- `clean_puffer1.py` → substitute the existing `clean_puffer1.py`

Additionally, the following files are **new** and must be included:
- `breaking_symmetries.py`
- `coloring.py`

## Future Plan
We are actively working on improving this process.  
In the future, running breaking symmetries will be possible directly from the **command line**, without needing to replace files manually.

For example, the usage may look like:

```bash
python main.py --mode breaking_symmetries

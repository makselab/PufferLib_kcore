# Temporary Folder for Breaking Symmetries

This folder contains temporary files needed to run **breaking symmetries**.  

## Current Usage (Hard Method)
To run the breaking symmetries functionality, you need to **replace** the existing files in the repository with the versions provided in this folder:
- `demo.py` → substitute the existing `demo.py`
- `clean_pufferl.py` → substitute the existing `clean_pufferl.py`

Additionally, the following files are **new** and must be included:
- `breaking_symmetries.py`
- `coloring.py`

# Breaking Symmetries: Parameters

## Current Parameters
The current implementation uses the following fixed settings:

- `per_cover = 0.5` → 50% of the non-trivial fibers will be broken automatically.  
- Breaking starts **after 500 episodes**.  
- After that, breaking occurs **every 300 episodes**.  

## How to Run
After substituting the files, you can run breaking symmetries using the **classic command**:

```bash
python demo.py --mode train --vec multiprocessing --env beam_rider > log.out 2> log.err
```

## Next updates

- Running breaking symmetries will be possible directly from the **command line**, without file substitution.  
- Parameters such as `per_cover`, start episode, and frequency will no longer be fixed but **user-defined through input line options**.  

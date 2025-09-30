MODEL COLLAPSING AND SYMMETRY BREAKING IN PUFFERLIB AGENTS: STEP-BY-STEP GUIDE
=================================================

This guide walks through how to use the fibration-based model collapsing features 
built into PufferLib agents to compress neural network layers and compare performance 
before and after compression. You'll run two training sessions and visualize 
the effects of collapsing using fibrations. The underlying architectures are typically composed of CCNs, LSTMs, and linear layers


STEP 1: RUN BASELINE TRAINING
-----------------------------
Run the original training using:

    python demo.py --mode train --vec multiprocessing --env beam_rider > log.out 2> log.err

This command:
- Trains the agent in the "beam_rider" Atari environment
- Saves stdout logs to `log.out`
- Saves error logs to `log.err`
- Stores model checkpoints under a subdirectory in `experiments/`


STEP 2: RUN COLLAPSED TRAINING
------------------------------
Run a second training session with the collapse mechanism enabled:

    python demo.py --mode train --vec multiprocessing --env beam_rider --collapse 0.3 > log_collapsed.out 2> log_collapsed.err

The new input argument `--collapse` (a value between 0 and 2, e.g., 0.3) 
activates the fiber collapsing pipeline.

This does two things:

1. Collapse the model:
   - Tries to load `model_000050.pt` from the experiment directory.
   - If not found, it finds the closest available checkpoint (e.g., `model_000051.pt`).
   - It compresses the model by applying fiber clustering to:
     * Linear layer
     * LSTM layer
   - The collapsed model and fiber info are saved under:
     
         experiments/<experiment-folder>/collapsed_model_thr_<THRESHOLDx10>_<EPISODE>/

     For example:

         experiments/beam_rider-e3d19a76/collapsed_model_thr_03_000051/

     This directory contains:
     - model_000051_collapsed.pt
     - fibers.pkl

2. Resume training from the collapsed model:
   - Continues training up to episode 300
   - Stores the new training run in a generated subfolder within the collapse directory


STEP 3: VISUALIZE AND COMPARE RESULTS
-------------------------------------
Use the provided notebook:

    Collapsing_step_by_step_guide.ipynb

This notebook:
- Parses `log.out` and `log_collapsed.out` for episode return data
- Loads `fibers.pkl` to track clustering reductions
- Generates a plot comparing original vs collapsed training performance
  and shows fiber cluster reduction over time


LOG FILES SUMMARY
-----------------
- `log.out`                  → Original training output
- `log.err`                 → Original training errors
- `log_collapsed.out`       → Collapsed model training output
- `log_collapsed.err`       → Errors from collapsed training session


FOLDER STRUCTURE EXAMPLE
-------------------------
experiments/
├── beam_rider-e3d19a76/
│   ├── model_000050.pt
│   ├── log.out
│   └── collapsed_model_thr_03_000051/
│       ├── model_000051_collapsed.pt
│       ├── fibers.pkl
│       └── beam_rider-<run_id>/
│           ├── model_000300.pt
│           └── log_collapsed.out

The final results should be something like: 
![Collapse Diagram](https://github.com/makselab/PufferLib_kcore/blob/2.0/Collapse.png?raw=true)

DISCLAIMER
----------
- The --collapse feature must be implemented in your local fork of PufferLib.
- Choose meaningful thresholds (e.g., 0.2, 0.5, 1.0).
- Always verify logs and fiber output to validate results.

# Empowerment Lander

Codebase built on work from https://github.com/rddy/deepassist. 

To set up required packages, use `environment.yml` with conda or `requirements.txt` with pip.

To train simulated pilots, run `train_sim_pilots.py`.
To train simulated pilots and copilots, run `train_copilot.py` with the argument `--empowerment` for adjusting empowerment coefficient (0 for no empowerment).
This will also automatically run the cross evaluation tests. 

To replay rollouts, run `run_rollouts.py` and replace the copilots with the saved policies.

To play the game (human trials), run the script `human_exp.sh`. This runs through the 
scripts `run_scripts/human_solo.py`, which is the vanilla game for getting accustomed to the controls,
then `python run_scripts/human_emp.py --empowerment`, which trains a copilot using empowerment, 
then `python run_scripts/human_emp.py`, which trains a copilot without empowerment.


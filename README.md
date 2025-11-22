# AML4NLPMiniProjectGroupX
Members: Alexander Rode (arod), Rasmus Herskind (rher)

Dataset: https://huggingface.co/datasets/stanfordnlp/imdb

## How to run code
We use python version 3.11.14, and torch with CUDA

Recommend to use conda. Then create an environment like:
```
conda env create -f environment.yaml
```
If using pip, install dependencies:
```
pip install -r requirements.txt
```
Then run the cells in the .ipynb files under the experiments folder or run them with `python 'python_file_here.py'`


## Useful commands for slurm
```
sbatch: to enqueue new jobs
scancel: to cancel existing jobs
squeue: to see the queue and running jobs
squeue -u <user>: View the users jobstatuses
seff: to analyse the resources used on completed jobs
tail -f <file>: View the last part of a file dynamically
```

# Central problem, domain, data characteristics
Sentiment analysis, Binary classification (negative / positive)

# Central method: chosen architecture and training mechanisms, with a brief justification if non-standard


# Key experiments & results: present and explain results, e.g. in simple accuracy tables over error graphs up to visualisations of representations and/or edge cases – keep it crisp


# Discussion: summarise the most important results and lessons learned (what is good, what can be improved)



TODO: Delete useful slurm commands
TODO: Compare with bert-base-uncased
TODO: Hyper-parameter-tuning

# OPTIONAL
TODO: Compare with another transformer model (distilBert or roberta)
TODO: Qualitative Error Analysis
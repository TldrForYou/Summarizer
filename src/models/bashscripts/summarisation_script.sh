#!/bin/bash

#SBATCH --partition=alpha
#SBATCH --nodes=1          
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4 
#SBATCH --mem=200000
#SBATCH --time=8:00:00
#SBATCH --account=p_scads_nlp
#SBATCH -e error-%j.err
#SBATCH -o out-%j.out

export WANDB_API_KEY="8b7b1701dfe5526a55a654facba19ff2a29a94a3"
export WANDB_ENTITY="du-pa"
export WANDB_PROJECT="summarisation_bot"


ml modenv/hiera GCC/10.3.0 CUDA/11.4.2 OpenMPI/4.1.1 Python git-lfs

source /scratch/ws/0/anpo879a-env_storage/1/venv_environment/python-environments/alpha_cu_11_4_py_3_9_5_gpt2_gen/bin/activate

python /beegfs/ws/0/anpo879a-beegfs_work/1/work_scads/MLops/Summarizer/src/features/summarization_train.py --model_name_or_path "google/bigbird-pegasus-large-pubmed" --tokenizer_name "google/bigbird-pegasus-large-pubmed" --use_auth_token True --dataset_name xsum --output_dir output/ --use_fast_tokenizer True --cache_dir "/scratch/ws/0/anpo879a-work_scads/anpo879a-work_scads-1661299321/work_scads/hf_cache" --overwrite_cache True --do_train --do_eval

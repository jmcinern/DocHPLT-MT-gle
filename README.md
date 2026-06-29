# Repo to train an MT English -> Irish model primarily for LLM data augmentation

## Data

The English Irish paired document subsection of the HPLT corpus. No document level MT model trained from it.

Link: https://huggingface.co/datasets/HPLT/DocHPLT/viewer/en-ga/

## Baseline

- MarianMT format V2 model: https://huggingface.co/HPLT/translate-en-ga-v2.0-hplt
- Will have to convert to hf/pytorch I think

## Evalutation

- The Dolly-15K instruction tuning dataset, translaetd by Gemini 2.5 Pro. Local Path: D:\VS-code-projects\droughts\LoRA_Ga\LoRA_Ga\Dolly_15K_En_Ga.jsonl
-- Each line has an instruction and response pair in English and the translated Irish
-- Take a subset of 100 lines for speedy eval and eval with BLEU-4

## Training

- Train on LUMI, access from local with "ssh lumi"
-- scratch: /scratch/project_465002364
- Local path to lumi docs: D:\VS-code-projects\lumi-userguide\docs
- Example file for multi GPU node T5 training on LUMI: D:\VS-code-projects\Denorm\train\nanoT5\train_denorm.sh

**NB: **
1. Env workflow: container + sqsh  for libraries not in container
2. Make sure to set bindings
3. Store data in parquet files to avoid transferring lots of small files over lustre file system

## Workflow
- You use an interactive GPU node on LUMI to debug env config. issuea
- You use a subset of the full dataset to test training
- You version control using git, LUMI uses agent forwarding
- I edit the README.md as our shared working doc that you read from and I write to 

## 
Agent1: Models: Configuring models for training
Agent2: Data: Download training data to LUMI (parquet), transfer eval data to LUMI 

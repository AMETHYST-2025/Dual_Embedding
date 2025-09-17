# Dual Embedding



## Overview


![image](./graphical_abstract.png)
# Dual Embedding: A Novel Fine-Tuned Language Model Approach for Accurate Polymer Glass Transition Temperature Prediction

This repository contains the code and part of data used in the study:

**Paper Title:**
*Dual Embedding: A Novel Fine-Tuned Language Model Approach for Accurate Polymer Glass Transition Temperature Prediction*

## Project Structure

- **`Sentence_BERT/`**
  Contains the general code used to fine-tune the PolyBERT model according to a selected score.

- **Experiment folders**:
  There are four folders, each corresponding to a different experiment:
  - `hypothetical_polymers/`
  - `benchmark_polymers/`
  - `polyimides/`
  - `homopolymers/`

Each experiment folder may include:
- The datasets used, except for very large files like the [hypothetical polymer dataset](https://zenodo.org/records/7766806) (which is nearly 1GB).
- A data loader script to generate datasets for the fine-tuning process in `Sentence_BERT`.
- Model training scripts.
- Some figures, such as histograms and result visualizations.
- --
- The folder "Sentence_BERT" contains the file `PB_finetuning.py`, which is an example of the embeddings fine-tuning code.

---
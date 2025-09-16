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

Each experiment folder generally includes:
- The datasets used, except for very large files like the [hypothetical polymer dataset](https://zenodo.org/records/7766806) (which is nearly 1GB).
- A data loader script to generate datasets for the fine-tuning process in `Sentence_BERT`.
- Fine-tuned models corresponding to each experiment.
  Due to file size limitations, only README files describing these models are included in this supplementary material.
  Full fine-tuned models are available upon request. If the paper is accepted, they will be uploaded to a public repository (e.g., Zotero) according to the conference policy, which prohibits including links during the review process.
- Model training scripts.
- Some figures, such as histograms and result visualizations.

---
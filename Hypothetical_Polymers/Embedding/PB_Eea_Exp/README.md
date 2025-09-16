---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:19888
- loss:CombinedLoss
base_model: kuelumbus/polyBERT
widget:
- source_sentence: '[*]CC([*])(C)C(=O)N(CCCOC(C)CCCCCC)C1CCCCC1'
  sentences:
  - '[*]Cc1cc(C(=O)N2CCOCC2)sc1C[*]'
  - '[*]Oc1cccc(-c2ccccc2-c2cccc(C(=O)Nc3cccc([*])c3)c2)c1'
  - '[*]c1cccc(CCCCCCOc2nc3cc(-c4cccc(-c5nc6cc([*])ccc6[nH]5)c4)ccc3[nH]2)c1'
- source_sentence: '[*]CC([*])c1ccc(Br)cc1-c1ccc(Oc2ccc3cc(OC)ccc3c2)cc1'
  sentences:
  - '[*]OC(=O)CCCCCCCCCCc1ccc(-c2ccc(S(=O)(=O)Oc3ccc(S(=O)(=O)O[*])cc3)cc2)cc1'
  - '[*]CCCCCSSc1cccc(-c2cccc(-c3cccc(CCCCC[*])c3)c2C#N)c1'
  - '[*]Oc1ccc(NC(=O)CCCCCCCCC([*])=O)c(OC)c1'
- source_sentence: '[*]CCCCCCCCCSSc1cccc2c(C(=O)N3C(=O)c4ccc(SSC[*])cc4C3=O)cccc12'
  sentences:
  - '[*]C(=O)c1ccc(-c2ccc(CCc3ccc(C#N)cc3)c(-c3ccc([*])cc3)c2)cc1C'
  - '[*]c1ccc2c(c1)C(CCCCCC)(CCCCCC)c1cc(-c3ccc(Oc4ccc(CCCCCCCC)cc4-c4ccc5c(c4)C(CCCCCC)(CCCCCC)c4cc([*])ccc4-5)cc3)ccc1-2'
  - '[*]CC([*])(C)C(=O)OOc1ccc(-c2ccc(-c3ccccc3OCC(O)CO)cc2)cc1'
- source_sentence: '[*]c1ccc(-c2cc([*])c(-c3ccc(-c4ccc(CCCCCCCCCCCC)cc4)c(C)c3)cc2-c2ccccc2)cc1'
  sentences:
  - '[*]Oc1cccc(C2(c3ccccc3C3(C([*])=O)CCCCCCCCCCC3)CCCCCCCCCCC2)c1'
  - '[*]CCCCCCCOS(=O)(=O)c1ccc(-c2ccc3c(c2)C(=O)N(COC[*])C3=O)cc1'
  - '[*]OC1CCCN1C(=O)NC1CCCN1c1ccc([*])cc1'
- source_sentence: '[*]CCCCCOC(=O)OCc1cccc(C(=O)NCCC[*])c1'
  sentences:
  - '[*]CC([*])C1(C)CC(OC2(C)CC(CCCCCC)CC(C)(C)C2)CC(C)(C)C1'
  - '[*]c1ccc(-c2cc[n+](Cc3ccc4c(c3)C(=O)N(c3ccc(-c5ccc6c(c5)C(=O)N(c5ccc([*])cc5)C6=O)cc3)C4=O)cc2)cc1'
  - '[*]CCCOCCc1ccc(-c2ccc(NC(=O)CCC[*])cc2)cc1'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on kuelumbus/polyBERT

This is a [sentence-transformers] model finetuned from [kuelumbus/polyBERT]. It maps sentences & paragraphs to a 600-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [kuelumbus/polyBERT] <!-- at revision deaa98fb65a7bdfb537457d42f43bd468963f695 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 600 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation]
- **Repository:** [Sentence Transformers on GitHub]
- **Hugging Face:** [Sentence Transformers on Hugging Face]

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: DebertaV2Model 
  (1): Pooling({'word_embedding_dimension': 600, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '[*]CCCCCOC(=O)OCc1cccc(C(=O)NCCC[*])c1',
    '[*]CC([*])C1(C)CC(OC2(C)CC(CCCCCC)CC(C)(C)C2)CC(C)(C)C1',
    '[*]c1ccc(-c2cc[n+](Cc3ccc4c(c3)C(=O)N(c3ccc(-c5ccc6c(c5)C(=O)N(c5ccc([*])cc5)C6=O)cc3)C4=O)cc2)cc1',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 600]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 19,888 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          | label                              |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-----------------------------------|
  | type    | string                                                                              | string                                                                              | list                               |
  | details | <ul><li>min: 27 tokens</li><li>mean: 67.22 tokens</li><li>max: 160 tokens</li></ul> | <ul><li>min: 28 tokens</li><li>mean: 65.04 tokens</li><li>max: 160 tokens</li></ul> | <ul><li>size: 2 elements</li></ul> |
* Samples:
  | sentence_0                                                                                                    | sentence_1                                                                                            | label                                                 |
  |:--------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|:------------------------------------------------------|
  | <code>[*]COCOCc1ccc(SSc2ccc(SSC[*])cc2)cc1</code>                                                             | <code>[*]C(=O)c1sc(-c2sc(C(C)(C)C)cc2-c2ccc([Si](CCCC)(CCCC)[Si]([*])(CCCC)CCCC)s2)cc1C(C)(C)C</code> | <code>[0.9812210970812252, 0.916353034566509]</code>  |
  | <code>[*]Cc1ccc(-c2ccc(OP(=O)(N=Nc3ccc(-c4ccc(N=NP(=O)(OC)OC(=O)CCCCCCCCCCC(=O)O[*])cc4)cc3)OC)cc2)cc1</code> | <code>[*]Oc1cccc(C2CCC(c3ccc(-c4ccc(Cc5ccc([*])cc5)cc4)cc3)CC2)c1</code>                              | <code>[0.918358703879871, 0.6840188792879889]</code>  |
  | <code>[*]Oc1cc(-c2cc(N3C(=O)c4ccc([*])cc4C3=O)c(CCCCCCCCCCCC)s2)c(C(C)(C)C)s1</code>                          | <code>[*]c1cc(CCCCCCCCCCCC)c(-c2cc(C(=O)OC)sc2[*])s1</code>                                           | <code>[0.6888697513698943, 0.2352090583150608]</code> |
* Loss: <code>__main__.CombinedLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 5
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4023 | 500  | 0.0246        |
| 0.8045 | 1000 | 0.0057        |
| 1.2068 | 1500 | 0.0032        |
| 1.6090 | 2000 | 0.0026        |
| 2.0113 | 2500 | 0.002         |
| 2.4135 | 3000 | 0.0015        |
| 2.8158 | 3500 | 0.0014        |
| 3.2180 | 4000 | 0.0012        |
| 3.6203 | 4500 | 0.001         |
| 4.0225 | 5000 | 0.0009        |
| 4.4248 | 5500 | 0.0008        |
| 4.8270 | 6000 | 0.0008        |


### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.4.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

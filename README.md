# Analyzing Large Language Models' Capability in Location Prediction

Code and data for the LREC-COLING 2024 paper
[Analyzing Large Language Models' Capability in Location Prediction](https://aclanthology.org/2024.lrec-main.85/)
by Zhaomin Xiao, Yan Huang, and Eduardo Blanco.

The paper analyzes whether large language models (FLAN-T5, FLAN-UL2,
FLAN-Alpaca, and ChatGPT) can determine if the author of a tweet is located in
the U.S. city the tweet mentions when the tweet was published. It studies (a)
instruction finetuning vs. off-the-shelf models, (b) the number of exemplars in
the prompt (zero/one/five/ten-shot), and (c) whether the surrounding context
(tweets published by the same author before and after the target tweet) helps.

## Repository map

- **Just want the corpus?** → `data/`
- **Looking for the numbers behind a table in the paper?** → see
  [Finding the paper's numbers](#finding-the-papers-numbers)
- **Want to rerun the experiments?** → `scripts/` (see
  [Running the experiments](#running-the-experiments))

| Path | Contents |
|---|---|
| `data/` | the corpus, the train/test split, and the city list (see below) |
| `llm.py`, `utils/` | code of the LLM experiments (FLAN-T5, FLAN-UL2, FLAN-Alpaca and their non-instruction-tuned counterparts; PEFT/LoRA instruction finetuning for the `+` experiments) |
| `scripts/notune.sh` | commands for all configurations without instruction finetuning |
| `scripts/tune0.sh`–`tune3.sh` | commands for all instruction-finetuning configurations |
| `outputs/flan/responses/` | raw generations of the preserved instruction-finetuned runs, one JSON list per configuration |
| `outputs/flan/results/` | their evaluation metrics (sklearn `classification_report` dicts); each file's weighted-average F1 (`["weighted avg"]["f1-score"]`) is the corresponding entry of Table 1 of the paper |

File names encode the configuration as
`<model>_<context>_<exemplar>`: `<model>` with a trailing `+` means
instruction-finetuned; `<context>` is one of `target` / `early_target` /
`target_later` / `all` (see below); `<exemplar>` is `zero` / `one` / `five` /
`ten`-shot.

## Corpus

The corpus is the one introduced by
[Xiao et al. (IJCNLP-AACL 2023 Findings)](https://aclanthology.org/2023.findings-ijcnlp.13/)
([aacl2023_repo](https://github.com/zhaomin1995/aacl2023_repo)): 3,494
annotated instances, each consisting of seven tweets published in chronological
order by the same user — three earlier tweets, one target tweet mentioning a
U.S. city, and three later tweets. Annotations indicate whether the author was
located in the city when the target tweet was published.

`data/data.json` has one JSON object per line (6,540 lines; the 3,494 lines
that carry at least one `Answer.Q1_*` key form the final corpus — the loaders
filter on this). Fields of each object:

| Field | Meaning |
|---|---|
| `instance_id` | unique id (`<event>_<tweet id>`); the ids in `data/data_split` refer to this field |
| `event` | the U.S. holiday around which the tweets were collected (e.g. `thanksgiving2019`) |
| `anchor_location` | the U.S. city mentioned in the target tweet (`<loc>` in the prompt) |
| `anchor_tweettext`, `anchor_timestamp` | text and timestamp of the **target tweet** |
| `context8/9/10_tweettext`, `..._timestamp` | the three tweets published **before** the target tweet, in chronological order |
| `context11/12/13_tweettext`, `..._timestamp` | the three tweets published **after** the target tweet, in chronological order |
| `Answer.Q1_<annotator>` | that annotator's answer: is the author located in the city? (`Yes` / `No`) |
| `Answer.Q2_<annotator>` | that annotator's confidence in their answer |
| `adjudicated_label` | the gold label (`Yes` / `No`), adjudicated with [MACE](https://github.com/dirkhovy/MACE) |
| `*_url`, `*_jsonpath`, `*_imagepath` | historical bookkeeping fields from corpus construction; the referenced files are not distributed |

The four context settings of the paper map to the `input_content` argument and
to the file names as follows:

| Paper (Table 3) | Here | Model input |
|---|---|---|
| Target | `target` | target tweet only |
| Earlier + Target | `early_target` | three earlier tweets + target tweet |
| Target + Later | `target_later` | target tweet + three later tweets |
| All | `all` | all seven tweets |

## Finding the paper's numbers

| Paper | Where to look |
|---|---|
| **Table 1** — weighted-avg F1 of all four LLMs, 0/1/5/10-shot, with and without instruction finetuning | the preserved runs in `outputs/flan/` (each result file reproduces its Table 1 entry); the commands for every FLAN configuration are in `scripts/`. ChatGPT was queried through the OpenAI API (see the note under [Running the experiments](#running-the-experiments)) |
| **Table 2** — the prompt | built by `get_prompt()` in `utils/data.py` |
| **Table 3** — the four context settings at 5-shot, without finetuning | the `<context>` segment of the file names (see the mapping above); the commands for every context setting are in `scripts/notune.sh` |
| **Table 4** — qualitative example on context tweets | instance from `data/data.json` |
| **Table 5** — ablation of the instruction modifications (FLAN-UL2, target, 5-shot) | *w/o preprocess* = skip `clean_text()` in `utils/data.py`; *w/o enhance* = drop the "If the tweet is associated with advertisements or news reports …" sentence from `get_prompt()` |
| **Table 6** — most common errors of FLAN-UL2 (target tweets) | examples from `data/data.json` |

## Running the experiments

Run everything from the repository root. Make sure you have enough disk space
for the pretrained weights (5–40 GB per model, downloaded from HuggingFace),
e.g.:

```
python llm.py -data_dir data/ -experiment flan_t5 -input_content all -output_dir outputs/flan -cache_dir <hf_cache> -exemplar zero-shot
```

Appending `+` to the experiment name (e.g. `flan_t5+`) runs instruction
finetuning with PEFT/LoRA before inference. All commands used in the paper are
in `scripts/`. In the paper the experiments were run on 4× NVIDIA A100-80GB
GPUs and took about 4 days in total. Note that freshly generated files are
written without the `.json` suffix that the preserved files carry.

The ChatGPT results were obtained in Oct 2023 (`gpt-3.5-turbo`, few-shot and
instruction finetuning through the OpenAI API), so they may not be exactly
reproducible after model upgrades on the OpenAI side. Please also be mindful
that instruction finetuning with ChatGPT is costly: we spent more than $500 on
these experiments.

## Privacy and anonymization

- All MTurk WorkerIds (the `Answer.*_<WorkerId>` keys of `data/data.json`) have
  been replaced with consistent pseudonyms `annotator_0001`,
  `annotator_0002`, ... The same pseudonym refers to the same crowdworker
  across this repository,
  [aacl2023_repo](https://github.com/zhaomin1995/aacl2023_repo), and
  [coling2022_repo](https://github.com/zhaomin1995/coling2022_repo). The
  mapping to real WorkerIds is not distributed.
- The corpus contains at most seven tweets per user published within 90 days,
  and only tweet text and timestamps are included — no user objects or other
  metadata. For takedown requests, please contact the authors.

If you have any questions about the paper, code, or data, please contact me via
email: zhaominxiao@my.unt.edu

## Citation

If you use the code or data, please cite the paper below.

```bibtex
@inproceedings{xiao-etal-2024-analyzing,
    title = "Analyzing Large Language Models{'} Capability in Location Prediction",
    author = "Xiao, Zhaomin  and
      Huang, Yan  and
      Blanco, Eduardo",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.85",
    pages = "951--958",
}
```

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments
)
from peft import LoraConfig


def get_train_args(output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        bf16=True,
        learning_rate=1e-5,
        logging_steps=50,
        save_strategy="epoch",
        save_steps=50,
        save_total_limit=5,
        # num_train_epochs=1,
        max_steps=1200,
    )
    return training_args


def get_peft_config(peft_name='lora'):
    if peft_name == 'lora':
        peft_config = LoraConfig(lora_alpha=8, lora_dropout=0.1, r=4, task_type="CAUSAL_LM")
    elif peft_name == 'prefix-tuning':
        pass  # this is on the research agenda
    elif peft_name == 'prompt-tuning':
        pass  # this is on the research agenda
    else:
        raise ValueError("Please check the name of peft method")
    return peft_config


def get_model_and_tokenizer(experiment, cache_dir):
    if '+' in experiment:
        experiment = experiment.replace("+", "")
    if experiment == 'flan_ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/flan-ul2')
        model = AutoModelForSeq2SeqLM.from_pretrained(
            'google/flan-ul2',
            load_in_8bit=True,  # if we have enough GPU memory, we can set load_in_8bit as False
            device_map="auto",
            cache_dir=cache_dir
        )
    elif experiment == 'flan_t5':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl",
            device_map="auto",
            load_in_8bit=True,
            cache_dir=cache_dir
        )
    elif experiment == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained("google/ul2")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/ul2",
            load_in_8bit=True,
            device_map="auto",
            cache_dir=cache_dir
        )
    elif experiment == 't5':
        tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/t5-v1_1-xxl",
            load_in_8bit=True,
            device_map="auto",
            cache_dir=cache_dir
        )
    elif experiment == 'flan_alpaca':
        tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-xxl")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "declare-lab/flan-alpaca-xxl",
            load_in_8bit=True,
            device_map="auto",
            cache_dir=cache_dir
        )
    elif experiment == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
    else:
        raise ValueError("Please check the name of the experiment")
    return tokenizer, model


def prepare_model_for_training(model):
    """
    This function is for preparing the model loaded in 8bit but training in 32 float precision
    :param model:
    :return:
    """
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False

    return model


def prepare_model_for_inference(model):
    """
    This function is for preparing the model for inference
    :param model:
    :return:
    """
    for param in model.parameters():
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    return model


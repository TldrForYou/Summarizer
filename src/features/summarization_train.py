""" This file is for training and/or fine-tuning of a NLP model """

import os
import sys
import logging
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import torch

import datasets
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    PegasusForConditionalGeneration,
    AutoConfig,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
import transformers

from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments
from huggingface_hub import login

import nltk
import wandb
import mlflow

mlflow.set_tracking_uri("http://89.19.208.000:5000")
mlflow.set_experiment("test_finetune")
mlflow.transformers.autolog()

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://89.19.208.000:9000"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "False"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "True"



wandb.login()

#Login token for HF
login(token="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")

# Initialize Wandb
wandb.init()

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, "
            "tag name or commit id)."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from "
            "huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the "
            "tokenizers library) or not."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` ("
                "necessary to use this script"
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training
    and eval.
    """

    lang: Optional[str] = field(
        default=None, metadata={"help": "Language id for summarization."}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets "
            "library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics ("
                "rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) "
            "on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer"
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after "
                "tokenization. Sequences longer"
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after "
                "tokenization. Sequences longer"
                "than this will be truncated, sequences shorter will be padded. Will "
                "default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of "
                "``model.generate``, which is used"
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the "
                "maximum length in the batch. More"
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "training examples to this"
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "evaluation examples to this"
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training, validation, or test file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], (
                    "`train_file` should be a csv or a json" " file."
                )
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], (
                    "`validation_file` should be a csv or a" " json file."
                )
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], (
                    "`test_file` should be a csv or a json " "file."
                )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}

# Metric
metric = load_metric("rouge")


class PegasusDataset(torch.utils.data.Dataset):
    """Preprocessing of the dataset"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.Tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.Tensor(
            self.labels["input_ids"][idx]
        )  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels["input_ids"])  # len(self.labels)


def main():
    """One function to bring them all"""
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is "
                f"not empty."
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                f"avoid this behavior, change"
                "the `--output_dir` or add `--overwrite_output_dir` to train from "
                "scratch."
            )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level
            # at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device},"
            f" n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits "
            f"training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

    def preprocess_function(examples):
        """Function for a preprocessing"""
        inputs = ["summarise" + doc for doc in examples["document"]]
        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, truncation=True
        )

        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"],
                max_length=data_args.max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(eval_pred):
        """Compute "custom" metrics for the xsum dataset"""

        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def optuna_hp_space(trial):
        """Space of Hyper parameters for Optuna"""

        return {
            "learning_rate": trial.suggest_float("learning_rate", 2e-5, 2e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [2, 4, 8]
            ),
        }

    args = ORTTrainingArguments(
        model_args.model_name_or_path,
        evaluation_strategy="epoch",
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=3,
        fp16=False,
        push_to_hub=True,
        report_to="mlflow, wandb",  # change to your favourite tracking platform
        optim="adamw_ort_fused",  # Fused Adam optimizer implemented by ORT
    )
    print("1")
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=True,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = PegasusForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # For HP optimisation
    def model_init(trial):
        return model

    raw_datasets = load_dataset(
        data_args.dataset_name,
        # data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Condition for the mode
    if model_args.model_name_or_path == "google/pegasus-pubmed":
        fake_preds = ["hello there", "general kenobi"]
        fake_labels = ["hello there", "general kenobi"]
        metric.compute(predictions=fake_preds, references=fake_labels)
        tokenizer = tokenizer
        tokenizer("Hello, this one sentence!")
        with tokenizer.as_target_tokenizer():
            print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]

    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        if training_args.do_train:
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(
                desc="train dataset map pre-processing"
            ):
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )

        if training_args.do_eval:
            max_target_length = data_args.val_max_target_length
            eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            with training_args.main_process_first(
                desc="validation dataset map pre-processing"
            ):
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
            val_dataset = eval_dataset
    mlflow_callback = transformers.integrations.MLflowCallback()
    trainer = ORTTrainer(
        model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        model_init=model_init,
        callbacks=[mlflow_callback],
        compute_metrics=compute_metrics,
        feature="seq2seq-lm",
    )

    if model_args.model_name_or_path == "google/pegasus-pubmed":
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=5,
            # compute_objective=compute_objective,
        )
    else:
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(metric_key_prefix="eval")
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "summarization",
        }
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = (
                    f"{data_args.dataset_name} " f"{data_args.dataset_config_name}"
                )
            else:
                kwargs["dataset"] = data_args.dataset_name

        if data_args.lang is not None:
            kwargs["language"] = data_args.lang

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

        return results


if __name__ == "__main__":
    main()
    mlflow.end_run()
    wandb.finish()

from transformers import Trainer, TrainingArguments


train_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=2e-5
)


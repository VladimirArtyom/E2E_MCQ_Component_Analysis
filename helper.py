from pytorch_lightning import LightningModule
from transformers import T5ForConditionalGeneration, AdamW, AutoTokenizer
from torch import Tensor
from typing import Mapping


class Dummy(LightningModule):

    def __init__(this, model: T5ForConditionalGeneration,
                 new_token_len: int,
                 lr: float):
        super().__init__()
        token = "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"
        this.model = model
        this.model.resize_token_embeddings(new_token_len)
        this.lr = lr

    def forward(this, input_ids: Tensor, attention_mask: Tensor, labels: Tensor=None):
            output = this.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)

            return output.loss, output.logits

    def training_step(this, batch: Mapping[str, Tensor]):
        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]
        labels: Tensor = batch["labels"]
        loss, _ = this(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        this.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(this):
         return AdamW(this.parameters(), lr=this.lr)


class Generate():
    context_token: str = "<context>"
    sep_token: str = "<sep>"
    answer_token: str = "<answer>"
    question_token: str = "<question>"
    
    @classmethod
    def generate_qag(cls, model: Dummy, tokenizer, context: str, answer: str, max_length_tokenizer: int, **kwargs):
        source_encoding = tokenizer(
            '{} {} {} {}'.format(cls.answer_token, answer, cls.context_token, context),
            max_length=max_length_tokenizer,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        generated_ids = model.model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            **kwargs
        )

        preds = {
            tokenizer.decode(generated_id, skip_special_tokens=False,clean_up_tokenization=True) for generated_id in generated_ids
        }

        return ''.join(preds)

    @classmethod
    def generate_qg(cls, model: Dummy, tokenizer, context, max_length_tokenizer, **kwargs):
        source_encoding = tokenizer(
            '{} {}'.format(cls.context_token, context),
            max_length=max_length_tokenizer,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        generated_ids = model.model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            **kwargs
        )

        preds = {
            tokenizer.decode(generated_id, skip_special_tokens=False,clean_up_tokenization=True) for generated_id in generated_ids
        }

        return ''.join(preds)

    @classmethod
    def generate_dg(cls, model: Dummy,
                        tokenizer,
                        context: str,
                        question: str,
                        answer: str,
                        max_length_tokenizer,
                        **kwargs):
        source_encoding = tokenizer(
            '{} {} {} {} {} {}'.format(cls.answer_token, answer,
                                       cls.question_token, question,
                                       cls.context_token, context),
            max_length=max_length_tokenizer,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        generated_ids = model.model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            **kwargs
        )

        preds = {
            tokenizer.decode(generated_id, skip_special_tokens=False,clean_up_tokenization=True) for generated_id in generated_ids
        }

        return ''.join(preds)
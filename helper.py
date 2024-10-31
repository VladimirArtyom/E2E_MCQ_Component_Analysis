from pytorch_lightning import LightningModule
from transformers import T5ForConditionalGeneration, AdamW, AutoTokenizer, T5Tokenizer
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
    def generate_qg(cls, model: Dummy,
                     tokenizer, context: str,
                     answer: str, max_length_tokenizer: int,
                     device: str = "cuda",
                     **kwargs):
        source_encoding = tokenizer(
            '{} {} {} {}'.format(cls.answer_token, answer, cls.context_token, context),
            max_length=max_length_tokenizer,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        generated_ids = model.model.generate(
            input_ids=source_encoding["input_ids"].to(device),
            attention_mask=source_encoding["attention_mask"].to(device),
            **kwargs
        )

        preds = {
            tokenizer.decode(generated_id, skip_special_tokens=False,clean_up_tokenization=True) for generated_id in generated_ids
        }

        return ''.join(preds)

    @classmethod
    def generate_qag(cls, model: Dummy,
        tokenizer, context,
        max_length_tokenizer,
        device: str,
         **kwargs):
        source_encoding = tokenizer(
            '{} {}'.format(cls.context_token, context),
            max_length=max_length_tokenizer,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        generated_ids = model.model.generate(
            input_ids=source_encoding["input_ids"].to(device),
            attention_mask=source_encoding["attention_mask"].to(device),
            **kwargs
        )

        preds = {
            tokenizer.decode(generated_id, skip_special_tokens=False,clean_up_tokenization=True) for generated_id in generated_ids
        }

        return ''.join(preds)

    @classmethod 
    def generate_paraphase(cls, 
                           model: Dummy,
                           tokenizer,
                           question: str,
                           max_length_tokenizer: int = 256,
                           device: str = "cuda",
                           **kwargs):
        source_encoding = tokenizer(
            "paraphrase: {} </s>".format(question),
            max_length=max_length_tokenizer,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        try:
            generated_ids = model.model.generate(
                input_ids=source_encoding["input_ids"].to(device),
                attention_mask=source_encoding["attention_mask"].to(device),
                **kwargs
            )
            preds = {
                tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization=True) for generated_id in generated_ids
            }

            return "".join(preds)
        except: 
            return "<UNK>"
        
    @classmethod
    def generate_dg_1(cls,
                      model_dg_1: Dummy,
                      model_paraphrase: Dummy,
                      tokenizer_dg,
                      tokenizer_paraphrase,
                      context: str,
                      question: str,
                      answer: str,
                      max_length_tokenizer_dg: int,
                      max_length_tokenizer_paraphrase: int,
                      device: str = "cuda",
                      **kwargs):
        paraphrase_kwargs = kwargs.get("para_kwargs")
        dg_kwargs = kwargs.get("dg_1_kwargs")

        distractor_candidates= []
        distractor_1 = cls.generate_dg(model_dg_1,
                                    tokenizer_dg,
                                    context,
                                    question,
                                    answer,
                                    max_length_tokenizer_dg,
                                    device,
                                    **dg_kwargs)

        distractor_candidates.append(distractor_1)
        paraphase_questions = cls.generate_paraphase(
            model_paraphrase,
            tokenizer_paraphrase,
            question,
            max_length_tokenizer_paraphrase,
            device,
            **paraphrase_kwargs
        )

        for question_ in paraphase_questions:
            distractor = cls.generate_dg(
                model_dg_1,
                tokenizer_dg,
                context,
                question_,
                answer,
                max_length_tokenizer_dg,
                device,
                **dg_kwargs
            )

            distractor_candidates.append(distractor)

            ## For now just append to it, and return
        return distractor_candidates            

    @classmethod
    def generate_dg(cls, model: Dummy,
                        tokenizer,
                        context: str,
                        question: str,
                        answer: str,
                        max_length_tokenizer,
                        device: str = "cuda",
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

        try:
            generated_ids = model.model.generate(
                input_ids=source_encoding["input_ids"].to(device),
                attention_mask=source_encoding["attention_mask"].to(device),
                **kwargs
            )

            preds = {
                tokenizer.decode(generated_id, skip_special_tokens=False,clean_up_tokenization=True) for generated_id in generated_ids
            }

            return ''.join(preds)
        except:
            return "<UNK>"
"""
if __name__ == "__main__":
    context = "JAKSldjasldkajsldkajsdklasjdlkasjdlkasdjlsajdsalkdjsaldkasjdlksajdask"
    question = "apa itu apa ?"
    answer = "yes sir"
    kwargs_para = {
        "num_beams": 3,
        "top_p": 0.98,
        "top_k": 130,
        "num_return_sequences":6,
        "repetition_penalty":3.2,
        "temperature": 1.8,
        "max_length": 256,
        "early_stopping":True,
        "do_sample": True
}

kwargs_dg = {
    "num_beams": 3,
    "top_p": 0.98,
    "top_k": 120,
    "temperature": 1.2,
    "max_length": 512,
    "num_return_sequences": 1,
    "repetition_penalty": 1.5,
    "no_repeat_ngram_size": 2,
    "early_stopping":True,
    "do_sample": True
}

kwargs = {
    "para_kwargs": kwargs_para,
    "dg_1_kwargs": kwargs_dg
}

path_para = "Wikidepia/IndoT5-base-paraphrase"

path_dg = "VosLannack/Distractor_1_t5-small"

token = "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"
tokenizer_para = T5Tokenizer.from_pretrained(path_para, use_auth_token=token)
tokenizer_dg = T5Tokenizer.from_pretrained(path_dg, use_auth_token=token)

model_para = T5ForConditionalGeneration.from_pretrained(path_para,return_dict=True ,use_auth_token=token).to("cuda")
model_dg = T5ForConditionalGeneration.from_pretrained(path_dg, return_dict=True,use_auth_token=token).to("cuda")

dum_para = Dummy(model_para, len(tokenizer_para), 1e-5)
dum_dg = Dummy(model_dg, len(tokenizer_dg), 1e-5)

out = Generate.generate_dg_1(dum_dg, dum_para, tokenizer_dg,
                            tokenizer_para, context, question, answer,
                            512, 256, "cuda", **kwargs)

print(out)
"""
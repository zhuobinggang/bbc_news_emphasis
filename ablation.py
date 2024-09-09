from main import Sector_2024, encode_with_title_and_sentences_truncate, get_embeddings
import torch
import torch.nn as nn

def get_tokenizer_bert(model_name = 'bert-base-cased'):
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(model_name)

def get_model_bert(model_name = 'bert-base-cased'):
    from transformers import BertModel
    return BertModel.from_pretrained(model_name)

def get_untrained_bert_model_and_tokenizer(model_name = 'bert-base-cased'):
    return get_model_bert(model_name), get_tokenizer_bert(model_name)

class Sector_without_crf(Sector_2024):
    def init_hook(self):
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_size, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 2)
        )
        # 不初始化CRF
    def get_loss(self, item):
        labels = item['labels']
        binary_class_logits = self.forward(item) # size: (sentence_num, 2)
        labels = torch.LongTensor(labels).cuda() # size: (sentence_num)
        loss = self.loss_fn(binary_class_logits, labels)
        return loss
    def predict(self, item):
        binary_class_logits = self.forward(item) # size: (sentence_num, 2)
        return torch.argmax(binary_class_logits, dim=1) # size: (sentence_num)
    def set_name(self):
        self.name = 'roberta-base-title-on-crf-off'

class Sector_without_title(Sector_2024):
    def forward(self, item):
        sentences = item['sentences']  # 只使用句子
        title = item['title']
        token_ids, head_ids = encode_with_title_and_sentences_truncate(self.tokenizer, sentences, title, empty_title=True)
        embeddings = get_embeddings(self.bert, token_ids, head_ids)
        binary_class_logits = self.classifier(embeddings) # size: (sentence_num, 2)
        return binary_class_logits
    def set_name(self):
        self.name = 'roberta-base-title-off-crf-on'

class Sector_without_roberta(Sector_2024):
    def init_bert(self):
        bert, tokenizer = get_untrained_bert_model_and_tokenizer('bert-base-cased') # 区分大小写，和roberta-base保持一致
        self.bert = bert
        self.tokenizer = tokenizer
    def set_name(self):
        self.name = 'bert-base-title-on-crf-on'


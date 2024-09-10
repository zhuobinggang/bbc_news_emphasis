from main import Sector_2024, encode_with_title_and_sentences_truncate, get_embeddings
import torch
import torch.nn as nn
import ablation

def encode_without_title(tokenizer, sentences, title):
    return encode_with_title_and_sentences_truncate(tokenizer, sentences, title, empty_title=True)

def encode_with_title(tokenizer, sentences, title):
    return encode_with_title_and_sentences_truncate(tokenizer, sentences, title, empty_title=False)

# TODO: 确认使用的是bert，确认encode_with_title_and_sentences_truncate开启了empty_title，确认self.crf为空
class Sector_bert_vanilla(Sector_2024):
    def init_bert(self):
        # 使用bert而不是roberta
        bert, tokenizer = ablation.get_untrained_bert_model_and_tokenizer('bert-base-cased') # 区分大小写，和roberta-base保持一致
        self.bert = bert
        self.tokenizer = tokenizer
    def init_hook(self):
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_size, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 2)
        )
        self.crf = None # 不初始化CRF
    def forward(self, item): # 新增empty_title参数, 默认使用[PAD]填充title
        sentences = item['sentences']
        title = item['title']
        token_ids, head_ids = encode_without_title(self.tokenizer, sentences, title)
        embeddings = get_embeddings(self.bert, token_ids, head_ids)
        binary_class_logits = self.classifier(embeddings) # size: (sentence_num, 2)
        return binary_class_logits
    def get_loss(self, item):
        labels = item['labels']
        binary_class_logits = self.forward(item) # size: (sentence_num, 2)
        labels = torch.LongTensor(labels).cuda() # size: (sentence_num)
        loss = self.loss_fn(binary_class_logits, labels) # 不使用CRF而使用交叉熵
        return loss
    def predict(self, item):
        binary_class_logits = self.forward(item) # size: (sentence_num, 2)
        # 不使用crf.decode而是torch.argmax
        return torch.argmax(binary_class_logits, dim=1) # size: (sentence_num)
    def set_name(self):
        self.name = 'bert_vanilla'

Sector_bert_vanilla.__name__ = 'bert_vanilla'

# TODO: 确认使用的是bert，确认encode_with_title_and_sentences_truncate开启了empty_title，确认self.crf不为空且get_loss和predict都使用了crf
class Sector_bert_crf_on(Sector_bert_vanilla):
    def init_hook(self):
        from torchcrf import CRF
        self.classifier = nn.Sequential(  # 因为要同时判断多种1[sep]3, 2[sep]2, 3[sep]1, 所以多加一点复杂度
            nn.Linear(self.bert_size, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 2)
        )
        self.crf = CRF(2, batch_first=True)
    def get_loss(self, item):
        labels = item['labels']
        binary_class_logits = self.forward(item).unsqueeze(0) # size: (1, sentence_num, 2)
        labels = torch.LongTensor([labels]).cuda() # size: (1, sentence_num)
        loss = -self.crf(binary_class_logits, labels)
        return loss
    def predict(self, item):
        binary_class_logits = self.forward(item)
        return self.crf.decode(binary_class_logits.unsqueeze(0))[0]
    def set_name(self):
        self.name = 'bert_crf_on'

Sector_bert_crf_on.__name__ = 'bert_crf_on'

# TODO: 确认使用的是bert，确认encode_with_title_and_sentences_truncate关闭了empty_title，确认self.crf为空
class Sector_bert_title_on(Sector_bert_vanilla):
    def forward(self, item): # 新增empty_title参数, 默认不使用[PAD]填充title
        sentences = item['sentences']
        title = item['title']
        # 关闭empty_title
        token_ids, head_ids = encode_with_title(self.tokenizer, sentences, title)
        embeddings = get_embeddings(self.bert, token_ids, head_ids)
        binary_class_logits = self.classifier(embeddings)
        return binary_class_logits
    def set_name(self):
        self.name = 'bert_title_on'

Sector_bert_title_on.__name__ = 'bert_title_on'

class Sector_bert_title_on_crf_on(ablation.Sector_without_roberta):
    pass

Sector_bert_title_on_crf_on.__name__ = ablation.Sector_without_roberta.__name__

# TODO: 确认使用的是roberta而不是bert，确认encode_with_title_and_sentences_truncate开启了empty_title，确认self.crf为空
class Sector_roberta_vanilla(Sector_bert_vanilla): 
    def init_bert(self):
        from main import get_untrained_model_and_tokenizer
        bert, tokenizer = get_untrained_model_and_tokenizer('roberta-base')
        self.bert = bert
        self.tokenizer = tokenizer
    def set_name(self):
        self.name = 'roberta_vanilla'

Sector_roberta_vanilla.__name__ = 'roberta_vanilla'

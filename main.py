import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from torchcrf import CRF

def get_tokenizer():
    from transformers import RobertaTokenizer
    return RobertaTokenizer.from_pretrained('roberta-base')

def get_model():
    from transformers import RobertaModel
    return RobertaModel.from_pretrained('roberta-base')

def get_untrained_model_and_tokenizer():
    return get_model(), get_tokenizer()

def encode_by_title_and_sentences(roberta_tokenizer, title, sentences):
    token_ids = []
    head_ids = []
    # 格式化标题并添加到句子前面
    formatted_title = f'Title: {title}'
    sentences = [formatted_title] + sentences
    for sentence in sentences:
        # 编码句子并添加[CLS]标记
        encoded = roberta_tokenizer.encode(sentence, add_special_tokens=True)
        token_ids.extend(encoded)
        # 记录[CLS]的下标（不包括标题的下标）
        if sentence != formatted_title:
            head_ids.append(len(token_ids) - len(encoded))
    return token_ids, head_ids

def encode_by_title_and_sentences_truncate(roberta_tokenizer, title, sentences):
    token_ids = []
    head_ids = []
    formatted_title = f'Title: {title}'
    sentences = [formatted_title] + sentences
    max_tokens_per_sentence = 480 // len(sentences) # 480是roberta-base的最大长度减去标题的大概长度
    for sentence in sentences:
        encoded = roberta_tokenizer.encode(sentence, add_special_tokens=True)
        if len(encoded) > max_tokens_per_sentence:
            encoded = encoded[:max_tokens_per_sentence - 2] + roberta_tokenizer.encode('...</s>', add_special_tokens=False)  # 保留'...'的token
        token_ids.extend(encoded)
        if sentence != formatted_title:
            head_ids.append(len(token_ids) - len(encoded))
    return token_ids, head_ids

# return size: (sentence_num, 768)
def get_embeddings(model, token_ids, head_ids):
    # 获取嵌入
    outputs = model(torch.tensor(token_ids).unsqueeze(0).cuda())
    cls_embeddings = outputs.last_hidden_state[0, head_ids]
    return cls_embeddings

def get_similarity(embeddings1, embeddings2):
    # 计算余弦相似度， 暂时用不到
    return torch.cosine_similarity(embeddings1, embeddings2, dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Sector_2024(nn.Module):
    def __init__(self, learning_rate=2e-5, cuda=True, last_dim = 768):
        super().__init__()
        self.learning_rate = learning_rate
        self.bert_size = last_dim
        self.verbose = False
        self.init_bert()
        self.init_hook()
        self.opter = optim.AdamW(
            self.get_should_update(),
            self.learning_rate)
        if cuda:
            self.cuda()
        self.is_cuda = cuda
        self.BCE = nn.BCELoss()
        self.name = 'roberta-base-title-on-crf-on'

    def forward(self, item):
        title = item['title']
        sentences = item['sentences']
        token_ids, head_ids = encode_by_title_and_sentences_truncate(self.tokenizer, title, sentences)
        embeddings = get_embeddings(self.bert, token_ids, head_ids) # size: (sentence_num, 768)
        binary_class_logits = self.classifier(embeddings) # size: (sentence_num, 2)
        return binary_class_logits

    
    def init_bert(self):
        bert, tokenizer = get_untrained_model_and_tokenizer()
        self.bert = bert
        self.tokenizer = tokenizer

    def init_hook(self):
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

    def get_should_update(self):
        return self.parameters()
    
    def predict(self, item):
        binary_class_logits = self.forward(item)
        return self.crf.decode(binary_class_logits.unsqueeze(0))[0]

    def get_name(self):
        return self.name
    

# ==============================

def verify_dataset(tokenizer):
    from dataset_verify import final_dataset
    train_set, test_set = final_dataset()
    articles = train_set + test_set
    # 初始化计数器
    total_articles = 0
    total_tokens = 0
    exceeding_count = 0

    for article in articles:  # 直接遍历文章对象
        title = article['title']
        sentences = article['sentences']
        token_ids, _ = encode_by_title_and_sentences(tokenizer, title, sentences)
        
        total_articles += 1
        total_tokens += len(token_ids)
        # 统计超过500个token的文章
        if len(token_ids) > 500:
            exceeding_count += 1

    average_tokens = total_tokens / total_articles if total_articles > 0 else 0
    exceeding_ratio = exceeding_count / total_articles if total_articles > 0 else 0

    print(f"Total articles: {total_articles}")
    print(f"Articles exceeding 500 tokens: {exceeding_count} ({exceeding_ratio:.2%})")
    print(f"Average token count per article: {average_tokens:.2f}")

def get_exceeding_articles(tokenizer):
    from dataset_verify import final_dataset
    train_set, test_set = final_dataset()
    articles = train_set + test_set
    exceeding_articles = []
    for article in articles:  # 直接遍历文章对象
        title = article['title']
        sentences = article['sentences']
        token_ids, _ = encode_by_title_and_sentences(tokenizer, title, sentences)
        
        # 统计超过500个token的文章
        if len(token_ids) > 500:
            exceeding_articles.append(article)
    return exceeding_articles

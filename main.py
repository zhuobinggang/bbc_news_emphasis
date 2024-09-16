import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from torchcrf import CRF

def get_tokenizer(model_name = 'roberta-base'):
    from transformers import RobertaTokenizer
    return RobertaTokenizer.from_pretrained(model_name)

def get_model(model_name = 'roberta-base'):
    from transformers import RobertaModel
    return RobertaModel.from_pretrained(model_name)

def get_untrained_model_and_tokenizer(model_name = 'roberta-base'):
    return get_model(model_name), get_tokenizer(model_name)

# TODO: 需要和encode_with_title_and_sentences_truncate一起测试
def encode_without_title(tokenizer, sentences, truncate = True):
    token_ids = []
    head_ids = []
    max_tokens_per_sentence = 500 // len(sentences)
    for sentence in sentences:
        encoded = tokenizer.encode(sentence, add_special_tokens=False)
        if truncate and len(encoded) > (max_tokens_per_sentence - 2):
            encoded = encoded[:max_tokens_per_sentence - 2]
        encoded = [tokenizer.cls_token_id] + encoded + [tokenizer.sep_token_id]
        token_ids.extend(encoded)
        head_ids.append(len(token_ids) - len(encoded))
    if len(head_ids) != len(sentences):
        print(f"head_ids length: {len(head_ids)}, original_sentences_num: {len(sentences)}")
        print(f"sentences: {sentences}")
        raise ValueError("head_ids和original_sentences_num的长度不匹配，程序终止。")
    if token_ids[head_ids[0]] != tokenizer.cls_token_id:
        raise ValueError("token_ids的第一个元素不是cls_token_id，程序终止。")
    if token_ids[-1] != tokenizer.sep_token_id:
        raise ValueError("token_ids的最后一个元素不是sep_token_id，程序终止。")
    if head_ids[0] != 0:
        raise ValueError("head_ids的第一个元素不是0，程序终止。注：encode_without_title的head_ids的第一个元素应该是0。")
    return token_ids, head_ids

def encode_with_title_and_sentences_truncate(tokenizer, sentences, title, truncate = True, empty_title = False):
    token_ids = []
    head_ids = []
    original_sentences_num = len(sentences)
    title_tokens = tokenizer.encode(title, add_special_tokens=False)
    if not empty_title:
        title_tokens = [tokenizer.cls_token_id] + title_tokens + [tokenizer.sep_token_id]
    else:
        title_tokens = [tokenizer.cls_token_id] + [tokenizer.pad_token_id] * len(title_tokens) + [tokenizer.sep_token_id]
    token_ids += title_tokens
    max_tokens_per_sentence = (500 - len(title_tokens)) // len(sentences) # 500是roberta-base的大概长度
    for sentence in sentences:
        encoded = tokenizer.encode(sentence, add_special_tokens=False)
        if truncate and len(encoded) > (max_tokens_per_sentence - 2):
            encoded = encoded[:max_tokens_per_sentence - 2]
        encoded = [tokenizer.cls_token_id] + encoded + [tokenizer.sep_token_id]
        token_ids.extend(encoded)
        head_ids.append(len(token_ids) - len(encoded))
    # 修改的assert逻辑
    if len(head_ids) != original_sentences_num:
        print(f"head_ids length: {len(head_ids)}, original_sentences_num: {original_sentences_num}")
        print(f"sentences: {sentences}")
        raise ValueError("head_ids和original_sentences_num的长度不匹配，程序终止。")
    if token_ids[head_ids[0]] != tokenizer.cls_token_id:
        raise ValueError("token_ids的第一个元素不是cls_token_id，程序终止。")
    if token_ids[-1] != tokenizer.sep_token_id:
        raise ValueError("token_ids的最后一个元素不是sep_token_id，程序终止。")
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
        self.loss_fn = nn.CrossEntropyLoss()
        self.set_name()

    def set_name(self):
        self.name = 'roberta-base-title-on-crf-on'

    def forward(self, item):
        title = item['title']
        sentences = item['sentences']
        token_ids, head_ids = encode_with_title_and_sentences_truncate(self.tokenizer, sentences, title)
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

Sector_2024.__name__ = 'roberta-base-title-on-crf-on'

# ==============================

def verify_dataset(tokenizer = get_tokenizer()):
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
        token_ids, _ = encode_with_title_and_sentences_truncate(tokenizer, sentences, title, truncate=False)
        
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

def get_exceeding_articles(tokenizer = get_tokenizer()):
    from dataset_verify import final_dataset
    train_set, test_set = final_dataset()
    articles = train_set + test_set
    exceeding_articles = []
    for article in articles:  # 直接遍历文章对象
        title = article['title']
        sentences = article['sentences']
        token_ids, _ = encode_with_title_and_sentences_truncate(tokenizer, sentences, title, truncate=False)
        
        # 统计超过500个token的文章
        if len(token_ids) > 500:
            exceeding_articles.append(article)
    return exceeding_articles

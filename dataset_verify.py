import os
import nltk
import random

def read_filepaths_by_category(base_path='./News Articles'):
    category_dict = {}
    
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            txt_files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith('.txt')]
            category_dict[category] = txt_files
    
    return category_dict

def get_articles():
    return read_filepaths_by_category(base_path='./News Articles')

def get_summaries():
    return read_filepaths_by_category(base_path='./Summaries')

def get_articles_and_summaries():
    articles = get_articles()
    summaries = get_summaries()
    return articles, summaries

def get_articles_and_summaries_by_category(category):
    # 修改为直接调用已有的函数
    articles = get_articles()[category]
    summaries = get_summaries()[category]
    return articles, summaries

def read_article(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:  # 尝试使用其他编码
            content = file.read()
    
    title, content = content.split('\n', 1)  # 直接分割为标题和内容
    sentences = [sentence.strip() for sentence in nltk.sent_tokenize(content)]  # 分句并去除前后空白
    
    # 处理包含换行的句子
    new_sentences = []
    for sentence in sentences:
        if '\n' in sentence:
            new_sentences.extend([s.strip() for s in sentence.split('\n')])  # 拆分并去除前后空白
        else:
            new_sentences.append(sentence)  # 直接添加句子
    
    return {
        'title': title.strip(),  # 使用提取的标题并去除前后空白
        'sentences': new_sentences  # 剩下的句子
    }

def read_summarization(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()  # 直接返回文件内容为字符串

def text_sub_matched(text, sentences):
    for sentence in sentences:
        text = text.replace(sentence, '')  # 删除存在的句子
    return text.strip()  # 返回去除前后空白的文本

# Total error count: 16
def data_verify():
    articles, summaries = get_articles_and_summaries()
    
    error_count = 0
    for category in articles.keys():
        for index, article_path in enumerate(articles[category]):
            article_data = read_article(article_path)
            summarization_text = read_summarization(summaries[category][index])
            
            # 调用text_sub_matched函数
            verified_text = text_sub_matched(summarization_text, article_data['sentences'])
            if verified_text:  # 如果返回的文本不为空
                print(f"Category: {category}, Article Path: {article_path}, Verified Text: {verified_text}")
                error_count += 1
    
    print(f"Total error count: {error_count}")

def read_dataset():
    articles, summaries = get_articles_and_summaries()
    dataset = []  # 用于存储最终的文章数据

    for category in articles.keys():
        for index, article_path in enumerate(articles[category]):
            article_data = read_article(article_path)
            summarization_text = read_summarization(summaries[category][index])
            
            # 调用text_sub_matched函数确保数据可靠
            verified_text = text_sub_matched(summarization_text, article_data['sentences'])
            if verified_text != '':  # 如果返回的文本不为空，跳过
                continue
            
            # 动态生成labels
            labels = [sentence in summarization_text for sentence in article_data['sentences']]
            
            # 构建文章对象并添加到dataset
            article_object = {
                'title': article_data['title'],
                'sentences': article_data['sentences'],
                'labels': labels,
                'category': category,
                'url': article_path  # 假设url为文章路径
            }
            dataset.append(article_object)

    return dataset  # 返回最终的文章数据集

def shuffled_articles(articles, seed=42):  # 新增函数
    random.seed(seed)  # 设置随机种子
    random.shuffle(articles)  # 直接打乱数组
    return articles  # 返回打乱后的文章数组

def five_folds_divided_dataset(articles, fold_index):  # 新增函数
    total_articles = len(articles)
    fold_size = total_articles // 5
    remainder = total_articles % 5
    # 计算每个fold的起始和结束索引
    test_start = fold_index * fold_size
    test_end = test_start + fold_size + (remainder if fold_index == 4 else 0)
    test_set = articles[test_start:test_end]  # 测试集
    train_set = articles[:test_start] + articles[test_end:]  # 训练集
    return train_set, test_set  # 返回训练集和测试集


def final_dataset(shuffle = True, fold_index = 0):
    articles = read_dataset()
    if shuffle:
        articles = shuffled_articles(articles)
    train_set, test_set = five_folds_divided_dataset(articles, fold_index)
    return train_set, test_set

if __name__ == "__main__":
    data_verify()
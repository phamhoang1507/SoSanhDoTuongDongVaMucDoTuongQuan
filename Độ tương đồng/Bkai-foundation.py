from scipy.stats import pearsonr, spearmanr

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


# Mean Pooling - Sử dụng attention mask để tính trung bình đúng
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Phần tử đầu tiên của model_output chứa tất cả các token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def similarity(text1, text2):
    # Tải mô hình từ HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
    model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

    # Token hóa các câu
    encoded_input = tokenizer(text1, padding=True, truncation=True, return_tensors='pt')
    encoded_input2 = tokenizer(text2, padding=True, truncation=True, return_tensors='pt')

    # Tính toán các token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        model_output2 = model(**encoded_input2)

    # Thực hiện pooling. Trong trường hợp này, sử dụng mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])

    embedding1 = sentence_embeddings[0].unsqueeze(0)
    embedding2 = sentence_embeddings2[0].unsqueeze(0)

    # Tính toán độ tương đồng cosine
    similarity = cosine_similarity(embedding1, embedding2)[0][0]

    return similarity.item()

txt1="Thật kinh ngạc, 42 khối vuông hình sắc cạnh không hơn kém nhau 1 gam."
txt2="Thật tuyệt vời, các khối hình đều giống nhau giống nhau một cách đáng kinh ngạc."


result = similarity(txt1, txt2)
print(result)

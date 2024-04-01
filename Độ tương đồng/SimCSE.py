import torch
from scipy.stats._mstats_basic import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize

def similarity(text1, text2):
    # Tokenize và chuyển đổi văn bản thành các vector embedding
    tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

    # Tokenize văn bản
    tokens1 = tokenizer(tokenize(text1), return_tensors="pt")
    tokens2 = tokenizer(tokenize(text2), return_tensors="pt")

    # Tính toán embedding
    with torch.no_grad():
        embeddings1 = model(**tokens1).pooler_output
        embeddings2 = model(**tokens2).pooler_output

    # Tính toán độ tương đồng sử dụng cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

    return cosine_sim.item()

txt1="Thật kinh ngạc, 42 khối vuông hình sắc cạnh không hơn kém nhau 1 gam."
txt2="Thật tuyệt vời, các khối hình đều giống nhau giống nhau một cách đáng kinh ngạc."

result = similarity(txt1, txt2)
print(result)

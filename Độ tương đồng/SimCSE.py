import torch
from scipy.stats._mstats_basic import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize

def similarity(sen1, sen2):
    # Tokenize và chuyển đổi văn bản thành các vector embedding
    tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

    # Tokenize văn bản
    tokens1 = tokenizer(tokenize(sen1), return_tensors="pt")
    tokens2 = tokenizer(tokenize(sen2), return_tensors="pt")

    # Tính toán embedding
    with torch.no_grad():
        embeddings1 = model(**tokens1).pooler_output
        embeddings2 = model(**tokens2).pooler_output

    # Tính toán độ tương đồng sử dụng cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

    return cosine_sim.item()

sen1="Thật kinh ngạc, 42 khối vuông hình sắc cạnh không hơn kém nhau 1 gam."
sen2="Thật tuyệt vời, các khối hình đều giống nhau giống nhau một cách đáng kinh ngạc."

result = similarity(sen1, sen2)

print("Câu 1: " ,sen1)
print("Câu 2: " ,sen2)
print("Độ tương đồng: ",result)

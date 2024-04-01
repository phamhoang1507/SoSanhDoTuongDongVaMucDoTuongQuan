from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
from difflib import SequenceMatcher

# Tải pre-trained model và tokenizer
model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Hàm để chuyển đoạn văn thành vector
def get_vector(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = phobert(input_ids)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector

txt1="Thật kinh ngạc, 42 khối vuông hình sắc cạnh không hơn kém nhau 1 gam."
txt2="Thật tuyệt vời, các khối hình đều giống nhau giống nhau một cách đáng kinh ngạc."
vector1 = get_vector(txt1)
vector2 = get_vector(txt2)
similarity = cosine_similarity([vector1], [vector2])[0][0]

print(similarity)

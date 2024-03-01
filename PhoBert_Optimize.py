from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
from difflib import SequenceMatcher


def similarity(text1, text2):
    # Tạo một đối tượng SequenceMatcher
    seq_matcher = SequenceMatcher(None, text1, text2)

    # Tính toán độ tương đồng
    similarity_ratio = seq_matcher.ratio()

    return similarity_ratio

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


f = open("DuLieuChuyenGia.txt", mode="r", encoding="utf-8")
simchuyengia = []
simtheobert = []
lst1 = []
lst2 = []
for s in f:
    s = s.replace('"', '')
    ss = s.split('\t')
    if (len(ss) != 5):
        print(s)
        continue
    temp = int(ss[3]) / 4.0
    vanban1 = ss[1]
    vanban2 = ss[2]
    # with  open("list1.txt", "a", encoding="utf-8") as f:
    #     f.write(vanban1 + "\n \n")
    # with  open("list2.txt", "a", encoding="utf-8") as f:
    #     f.write(vanban2 + "\n \n")

    lst1.append(vanban1)
    lst2.append(vanban2)
    simchuyengia.append(temp)
f.close()

# Tạo vector cho mỗi đoạn văn bản
for i in range(len(lst1)):
    vector1 = get_vector(lst1[i])
    vector2 = get_vector(lst2[i])
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    simtheobert.append(similarity)

optimizer = torch.optim.AdamW(phobert.parameters(), lr=2e-5)
loss_fn = torch.nn.MSELoss()

# Fine-tuning loop
num_epochs = 5
batch_size = 16

for epoch in range(num_epochs):
    for i in range(0, len(lst1), batch_size):
        batch1 = lst1[i:i+batch_size]
        batch2 = lst2[i:i+batch_size]
        labels = torch.tensor(simchuyengia[i:i+batch_size], dtype=torch.float32)

        # Tokenize and create input tensors
        encoded_input1 = tokenizer(batch1, padding=True, truncation=True, return_tensors='pt')
        encoded_input2 = tokenizer(batch2, padding=True, truncation=True, return_tensors='pt')

        # Forward pass
        output1 = phobert(**encoded_input1).last_hidden_state.mean(dim=1).squeeze()
        output2 = phobert(**encoded_input2).last_hidden_state.mean(dim=1).squeeze()

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(output1, output2).flatten()

        # Compute loss
        loss = loss_fn(similarity_scores, labels)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

pearson = pearsonr(simtheobert, simchuyengia)
spearman = spearmanr(simtheobert, simchuyengia)
print(simtheobert)
print(simchuyengia)
print("Tương Quan Theo Phương Pháp PearSon : ", pearson[0])
print("Tương Quan Theo Phương Pháp SpearMan : ", spearman[0])

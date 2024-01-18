from asynchat import simple_producer
from difflib import SequenceMatcher
import torch
from torch import cosine_similarity

from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize
import numpy as np

def similarity(text1, text2):
    # Tạo một đối tượng SequenceMatcher
    seq_matcher = SequenceMatcher(None, text1, text2)

    # Tính toán độ tương đồng
    similarity_ratio = seq_matcher.ratio()

    return similarity_ratio

PhobertTokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

lst1=[]
lst2=[]
simchuyengia = []
simtheobert = []
f = open("data1.txt", mode="r", encoding="utf-8")

for s in f:
    s = s.replace('"', '')
    ss = s.split('\t')
    if (len(ss) != 5):
        print(s)
        continue
    temp = int(ss[3]) / 4.0
    vanban1 = ss[1]
    vanban2 = ss[2]

    lst1.append(vanban1)
    lst2.append(vanban2)
    simchuyengia.append(temp)
f.close()


lst1_sentences = [tokenize(sentence) for sentence in lst1]
lst2_sentences = [tokenize(sentence) for sentence in lst2]

lst1_inputs = PhobertTokenizer(lst1_sentences, padding=True, truncation=True, return_tensors="pt")
lst2_inputs = PhobertTokenizer(lst2_sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    lst1_embeddings = model(**lst1_inputs, output_hidden_states=True, return_dict=True).pooler_output
    lst2_embeddings = model(**lst2_inputs, output_hidden_states=True, return_dict=True).pooler_output
    # print(lst1_embeddings)
    # print(lst2_embeddings)

lst1_np = lst1_embeddings.numpy()
lst2_np = lst2_embeddings.numpy()
print(lst1_np)
print(lst2_np)
for i in range(len(lst1_np)):
    dotuongdong = similarity(lst1_np[i],lst2_np[i])
    print(dotuongdong)
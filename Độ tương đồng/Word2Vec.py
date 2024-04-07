from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def word2vec(sen1,sen2):
    tokens1 = sen1.split()
    tokens2 = sen2.split()

    # Xây dựng mô hình Word2Vec
    model = Word2Vec([tokens1, tokens2], vector_size=100, window=5, min_count=1, workers=4)

    # Lấy vectơ biểu diễn của câu
    vector1 = np.mean([model.wv[word] for word in tokens1], axis=0)
    vector2 = np.mean([model.wv[word] for word in tokens2], axis=0)

    # Tính mức độ tương đồng sử dụng cosine similarity
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity

sen1="Thật kinh ngạc, 42 khối vuông hình sắc cạnh không hơn kém nhau 1 gam."
sen2="Thật tuyệt vời, các khối hình đều giống nhau giống nhau một cách đáng kinh ngạc."

simWord2Vec= word2vec(sen1,sen2)
print("Câu 1: " ,sen1)
print("Câu 2: " ,sen2)
print("Độ tương đồng: ",simWord2Vec)
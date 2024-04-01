import numpy as np
from astropy.stats import spatial

data="D:\\Demo\\code_python\\W2V_150.txt" #file word embedding
dim=150
def load_mappings(path):
    # save word mapping dictionary to pkl file for quickly load for further use
    word_mapping = dict()
    with open(path, 'r',encoding="utf-8") as f:
        # skip first two lines which are vocab size and embedding dimension
        for line in f.readlines():
            if len(line.split())<3:
            	continue
            word, vec = line.split(' ', 1)                        # split word and vector
            vec = np.fromstring(vec, sep=' ')                      # load str to np.array
            word=word.strip()
            word=word.replace(" ","_")
            #if word not in words:
                #words.append(word)
            word_mapping[word] = vec                               # append word:vec to dictionary
    return word_mapping
def getVectors(u1):
    u1 = u1.replace("-","_")
    vs=np.repeat(0.0001,dim)
    mk=1
    if (u1 in model.keys()):
        vs = model[u1]
    else:
        uu=u1.split("_")
        for i in uu:
            if (i in model.keys()):
                if (mk==1):
                    vs=model[i]
                    mk=0
                else:
                    vs=vs+model[i]
    return  vs

model=load_mappings(data)
#tuong tu giua 2 tu
def sim2word(word1,word2):
    if word1 in model:
        v1 = model[word1]
    else:
        v1= getVectors(word1.strip())
    if word2 in model:
        v2 = model[word2]
    else:
        v2= getVectors(word2.strip())
    return ((2 - spatial.distance.cosine(v1, v2))/2)
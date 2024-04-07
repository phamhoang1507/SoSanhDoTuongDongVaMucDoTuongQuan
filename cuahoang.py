from tkinter import *
from tkinter import ttk, messagebox, scrolledtext

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import torch

from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize
win = Tk()
s = ttk.Style()
s.theme_use('clam')
win.title('Phần Mềm So Sánh Mức độ tương đồng giữa hai văn bản')
win.geometry('950x470')
win['bg'] = 'grey'
win.attributes('-topmost', True)

def clear_labels_and_progressbars():
    # Xóa các Label và Progressbar
    for widget in win.winfo_children():
        if isinstance(widget, Label) and widget.winfo_y() >= 296:
            widget.destroy()
        elif isinstance(widget, ttk.Progressbar):
            widget.destroy()

def on_enter(event):
    s.map("btnss.TButton", background=[("active", '#33CC66')])
    btnss.config(cursor="hand2")
    s.map("btncl.TButton", background=[("active", '#FF3333')])
    btncl.config(cursor="hand2")
    s.map("btnmf.TButton", background=[("active", '#FF9966')])
    btnmf.config(cursor="hand2")

lb1 = Label(win, text="Văn Bản Thứ Nhất", font=('Times New Roman', 12), bg='grey', fg='white')
lb1.place(x=30, y=30)

text1 = Text(win, width=80, height=3, font=('Times New Roman', 12))
text1.place(x=30, y=60)
text1.focus()

def openf():
    f = open("bodulieuVanBan.json", mode="r", encoding="utf-8")
    content = f.read()
    showContent(content)

def showContent(content):
    new_window = Tk()
    new_window.title('Nội dung tệp dữ liệu')
    new_window.geometry("+1150+170")
    text_widget = scrolledtext.ScrolledText(new_window, width=80, height=20, wrap=WORD)
    text_widget.insert(END, content)
    text_widget.pack(expand=True, fill='both')

    new_window.mainloop()


s.configure("btnmf.TButton",
            foreground="white",
            background="#FF9900",
            padding=(55, 5),
            font=('Times New Roman', 14),
            borderwidth=2,
            relief="solid",
            bordercolor ="#FF9966"
            )
btnmf = ttk.Button(win, text="Mở file văn bản", style="btnmf.TButton",command=openf)
btnmf.place(x=700, y=60)
btnmf.bind("<Enter>", on_enter)

lb2 = Label(win, text="Văn Bản Thứ Hai", font=('Times New Roman', 12), fg='white')
lb2['bg'] = 'grey'
lb2.place(x=30, y=150)

text2 = Text(win, width=80, height=3, font=('Times New Roman', 12))
text2.place(x=30, y=180)

# phobert-V2
model_phobert = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_phobert)
tokenizer_phobert = AutoTokenizer.from_pretrained(model_phobert)

listvb1 = []
listvb2 = []

def phobertV2(text1, text2):
    input_ids1 = tokenizer_phobert.encode(text1, return_tensors="pt")
    input_ids2 = tokenizer_phobert.encode(text2, return_tensors="pt")
    with torch.no_grad():
        outputs1 = phobert(input_ids1)
        outputs2 = phobert(input_ids2)
    vector1 = outputs1.last_hidden_state.mean(dim=1).squeeze().numpy()
    vector2 = outputs2.last_hidden_state.mean(dim=1).squeeze().numpy()
    return cosine_similarity([vector1], [vector2])[0][0]

# SimCSE
tokenizer_SimCSE = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
model_SimCSE = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

def simCSE_Phobert(text1, text2):

    tokens1 = tokenizer_SimCSE(tokenize(text1), return_tensors="pt")
    tokens2 = tokenizer_SimCSE(tokenize(text2), return_tensors="pt")

    # Tính toán embedding
    with torch.no_grad():
        embeddings1 = model_SimCSE(**tokens1).pooler_output
        embeddings2 = model_SimCSE(**tokens2).pooler_output

    # Tính toán độ tương đồng sử dụng cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

    return cosine_sim.item()

#bkai-foundation-model
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Phần tử đầu tiên của model_output chứa tất cả các token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
# Tải mô hình từ HuggingFace Hub
tokenizer_bkai = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
model_bkai = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

def bkai(text1, text2):

    # Token hóa các câu
    encoded_input = tokenizer_bkai(text1, padding=True, truncation=True, return_tensors='pt')
    encoded_input2 = tokenizer_bkai(text2, padding=True, truncation=True, return_tensors='pt')

    # Tính toán các token embeddings
    with torch.no_grad():
        model_output = model_bkai(**encoded_input)
        model_output2 = model_bkai(**encoded_input2)

    # Thực hiện pooling. Trong trường hợp này, sử dụng mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])

    embedding1 = sentence_embeddings[0].unsqueeze(0)
    embedding2 = sentence_embeddings2[0].unsqueeze(0)

    # Tính toán độ tương đồng cosine
    similarity = cosine_similarity(embedding1, embedding2)[0][0]

    return similarity.item()

#Word2Vec
def word2vec(text1,text2):
    tokens1 = text1.split()
    tokens2 = text2.split()

    # Xây dựng mô hình Word2Vec
    model = Word2Vec([tokens1, tokens2], vector_size=100, window=5, min_count=1, workers=4)

    # Lấy vectơ biểu diễn của câu
    vector1 = np.mean([model.wv[word] for word in tokens1], axis=0)
    vector2 = np.mean([model.wv[word] for word in tokens2], axis=0)

    # Tính mức độ tương đồng sử dụng cosine similarity
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity
def click():
    vb1 = text1.get("1.0", "end-1c")
    vb2 = text2.get("1.0", "end-1c")
    sentences1 = vb1.split('.')
    sentences2 = vb2.split('.')
    listvb1.clear()  # Xóa danh sách cũ
    listvb2.clear()  # Xóa danh sách cũ
    for i in sentences1:
        listvb1.append(i.strip())  # Thêm vào danh sách mới
    for j in sentences2:
        listvb2.append(j.strip())  # Thêm vào danh sách mới
    sopt1 = len(listvb1)
    sopt2 = len(listvb2)
    tongpho = 0
    tongsim = 0
    tongbkai = 0
    tongw2v = 0
    for i in listvb1:
            maxsspho = 0
            maxsssim=0
            maxssbkai=0
            maxssw2v=0
            for j in listvb2:
                similaritypho = phobertV2(i, j)
                similaritysim = simCSE_Phobert(i, j)
                similaritybkai= bkai(i,j)
                similarityw2v=word2vec(i,j)
                if similaritypho > maxsspho:
                    maxsspho = similaritypho
                if similaritysim>maxsssim:
                    maxsssim=similaritypho
                if similaritybkai>maxssbkai:
                    maxssbkai=similaritybkai
                if similarityw2v>maxssw2v:
                    maxssw2v=similarityw2v
            if maxsspho < 0.8:
                maxsspho = 0
            if maxsssim<0.8:
                maxsssim = 0
            if maxssbkai<0.8:
                maxssbkai=0
            if maxssw2v<0.8:
                maxssw2v=0

            tongpho += maxsspho
            tongsim +=maxsssim
            tongbkai+=maxssbkai
            tongw2v +=maxssw2v
    for j in listvb2:
        maxsspho = 0
        maxsssim = 0
        maxssbkai = 0
        maxssw2v = 0
        for i in listvb2:
            similaritypho = phobertV2(i, j)
            similaritysim = simCSE_Phobert(i, j)
            similaritybkai = bkai(i, j)
            similarityw2v = word2vec(i, j)
            if similaritypho > maxsspho:
                maxsspho = similaritypho
            if similaritysim > maxsssim:
                maxsssim = similaritysim
            if similaritybkai > maxssbkai:
                maxssbkai = similaritybkai
            if similarityw2v > maxssw2v:
                maxssw2v = similarityw2v
        if maxsspho < 0.8:
            maxsspho = 0
        if maxsssim < 0.8:
            maxsssim = 0
        if maxssbkai < 0.8:
            maxssbkai = 0
        if maxssw2v < 0.8:
            maxssw2v = 0

        tongsim  += maxsssim
        tongbkai += maxssbkai
        tongw2v  += maxssw2v

    # tongsim= tongsim/(sopt1+sopt2)
    # tongbkai =tongbkai /(sopt1+sopt2)
    tongw2v = tongw2v / (sopt1+sopt2)
    print(tongw2v)

    # def update_progress2(value):
    #     if vb1.strip() != "" and vb2.strip() != "":
    #         bar_simcse['value'] = value
    #
    #         if value < tongsim * 100:
    #             win.after(20, update_progress2, value + 1)
    #
    #             if tongsim < 0.6:
    #                 s.configure("sim.Horizontal.TProgressbar", foreground='red', background='red')
    #                 lb5 = Label(win, text="SimCSE: ", font=('Times New Roman', 14), bg="grey",
    #                             fg="white")
    #                 lb5.place(x=30, y=338)
    #
    #                 lb6 = Label(win, text=" " + str(tongsim), font=('Times New Roman', 14),
    #                             bg="grey",
    #                             fg="white")
    #                 lb6.place(x=490, y=339)
    #             elif tongsim >= 0.6 and tongsim < 0.8:
    #                 s.configure("sim.Horizontal.TProgressbar", foreground='red', background='yellow')
    #                 lb5 = Label(win, text="SimCSE: ", font=('Times New Roman', 14), bg="grey",
    #                             fg="white")
    #                 lb5.place(x=30, y=338)
    #
    #                 lb6 = Label(win, text=" " + str(tongsim), font=('Times New Roman', 14),
    #                             bg="grey",
    #                             fg="white")
    #                 lb6.place(x=490, y=339)
    #
    #             else:
    #                 s.configure("sim.Horizontal.TProgressbar", foreground='red', background='green')
    #                 lb5 = Label(win, text="SimCSE: ", font=('Times New Roman', 14), bg="grey",
    #                             fg="white")
    #                 lb5.place(x=30, y=338)
    #
    #                 lb6 = Label(win, text=" " + str(tongsim), font=('Times New Roman', 14),
    #                             bg="grey",
    #                             fg="white")
    #                 lb6.place(x=490, y=339)
    #             bar_simcse['value'] = tongsim * 100
    #
    #     else:
    #         messagebox.showwarning("Thông báo",
    #                                "Văn bản thứ nhất hoặc thứ hai đang bị trống. Vui lòng điền đầy đủ trước khi so sánh")
    #
    # bar_simcse = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
    #                       style='sim.Horizontal.TProgressbar')
    # bar_simcse.place(x=180, y=340, height="25")
    # update_progress2(0)

    # def update_progress3(value):
    #     if vb1 != "" and vb2.strip() != "":
    #         bar_bkai['value'] = value
    #         if value < tongbkai * 100:
    #             win.after(20, update_progress3, value + 1)
    #             if tongbkai <0.6:
    #                 s.configure("bkai.Horizontal.TProgressbar", foreground='red', background='red')
    #                 lb5 = Label(win, text="SimBkai: ", font=('Times New Roman', 14), bg="grey",
    #                                 fg="white")
    #                 lb5.place(x=30, y=378)
    #
    #                 lb6 = Label(win, text=" " + str(tongbkai), font=('Times New Roman', 14),
    #                                 bg="grey",
    #                                 fg="white")
    #                 lb6.place(x=490, y=379)
    #             elif tongbkai >=0.6 and tongbkai <0.8:
    #                 s.configure("bkai.Horizontal.TProgressbar", foreground='red', background='yellow')
    #                 lb5 = Label(win, text="SimBkai: ", font=('Times New Roman', 14), bg="grey",
    #                                 fg="white")
    #                 lb5.place(x=30, y=378)
    #
    #                 lb6 = Label(win, text=" " + str(tongbkai), font=('Times New Roman', 14),
    #                                 bg="grey",
    #                                 fg="white")
    #                 lb6.place(x=490, y=379)
    #             else:
    #                 s.configure("bkai.Horizontal.TProgressbar", foreground='red', background='green')
    #                 lb5 = Label(win, text="SimBkai: ", font=('Times New Roman', 14), bg="grey",
    #                                 fg="white")
    #                 lb5.place(x=30, y=378)
    #
    #                 lb6 = Label(win, text=" " + str(tongbkai), font=('Times New Roman', 14),
    #                                 bg="grey",
    #                                 fg="white")
    #                 lb6.place(x=490, y=379)
    #             bar_bkai['value'] = tongbkai * 100
    # bar_bkai = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
    #                           style='bkai.Horizontal.TProgressbar')
    # bar_bkai.place(x=180, y=380,height="25")
    # update_progress3(0)

    def update_progress4(value):
        if vb1 != "" and vb2.strip() != "":
            bar_w2v['value'] = value
            if value < tongw2v * 100:
                win.after(20, update_progress4, value + 1)
                if tongw2v <0.6:
                    s.configure("w2v.Horizontal.TProgressbar", foreground='red', background='red')
                    lb5 = Label(win, text="Word2Vec: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=418)

                    lb6 = Label(win, text=" " + str(tongw2v), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=419)
                elif tongw2v >=0.6 and tongw2v <0.8:
                    s.configure("w2v.Horizontal.TProgressbar", foreground='red', background='yellow')
                    lb5 = Label(win, text="Word2Vec: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=418)

                    lb6 = Label(win, text=" " + str(tongw2v), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=419)
                else:
                    s.configure("w2v.Horizontal.TProgressbar", foreground='red', background='green')
                    lb5 = Label(win, text="Word2Vec: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=418)

                    lb6 = Label(win, text=" " + str(tongw2v), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=419)
                bar_w2v['value'] = tongw2v * 100
    bar_w2v = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
                              style='w2v.Horizontal.TProgressbar')
    bar_w2v.place(x=180, y=420,height="25")
    update_progress4(0)

s.configure("btnss.TButton",
            foreground="white",
            background="#00CC00",
            padding=(62, 5),
            font=('Times New Roman', 14),
            borderwidth=2,
            relief="solid",
            bordercolor ="#33CC66",
            )

btnss = ttk.Button(win, text="So sánh",command=click, style="btnss.TButton")
btnss.place(x=700, y=130)
btnss.bind("<Enter>", on_enter)

def clear():
    text1.delete(1.0, END)
    text1.focus()
    text2.delete(1.0, END)
    clear_labels_and_progressbars()

s.configure("btncl.TButton",
            foreground="white",
            background="#FF0000",
            padding=(63, 5),
            font=('Times New Roman', 14),
            borderwidth=2,
            relief="solid",
            bordercolor="#FF3333"
            )
btncl = ttk.Button(win, text="Làm mới", style="btncl.TButton",command=clear)
btncl.place(x=700, y=200)
btncl.bind("<Enter>", on_enter)

win.mainloop()

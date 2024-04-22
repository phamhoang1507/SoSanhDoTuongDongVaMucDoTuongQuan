import shutil
from tkinter import *
from tkinter import messagebox, ttk
from tkinter.ttk import Progressbar, Style
from ttkthemes import ThemedStyle
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize
from tkinter import scrolledtext
from gensim.models import Word2Vec
import numpy as np

def clear_labels_and_progressbars():
    # Xóa các Label và Progressbar
    for widget in win.winfo_children():
        if isinstance(widget, Label) and widget.winfo_y() >= 296:
            widget.destroy()
        elif isinstance(widget, ttk.Progressbar):
            widget.destroy()

win = Tk()
s = ttk.Style()
s.theme_use('clam')
win.title('Phần Mềm So Sánh Mức độ tương đồng giữa các câu')
win.geometry('970x520')
win['bg'] = 'grey'
win.attributes('-topmost', True)

lb1 = Label(win, text="Câu Thứ Nhất", font=('Times New Roman', 12), bg='grey', fg='white')
lb1.place(x=30, y=80)

lbtieude =Label(win,text="So Sánh Độ Tương Đồng",font=('Times New Roman', 40), bg='grey', fg='white')
lbtieude.place(x=200,y=5)

text1 = Text(win, width=80, height=4, font=('Times New Roman', 12))
text1.place(x=30, y=110)
text1.focus()

def on_enter(event):
    s.map("btnss.TButton", background=[("active", '#33CC66')])
    btnss.config(cursor="hand2")
    s.map("btnlm.TButton", background=[("active", '#FF3333')])
    btnlm.config(cursor="hand2")
    s.map("btnopen1.TButton", background=[("active", '#FF9966')])
    btnopen1.config(cursor="hand2")
    s.map("btnopen2.TButton", background=[("active", '#FF9966')])
    btnopen2.config(cursor="hand2")


def openfile1():
    f = open("../Bộ dữ liệu/list1.txt", mode="r", encoding="utf-8")
    content= f.read()
    showContent(content)

s.configure("btnopen1.TButton",
            foreground="white",
            background="#FF9900",
            padding=(16, 5),
            font=('Times New Roman', 14),
            borderwidth=2,
            relief="solid",
            bordercolor ="#FF9966"
            )
btnopen1 = ttk.Button(win, text="Mở file dữ liệu thứ nhất",command=openfile1, style="btnopen1.TButton")
btnopen1.place(x=720, y=110)
btnopen1.bind("<Enter>", on_enter)

lb2 = Label(win, text="Câu Thứ Hai", font=('Times New Roman', 12), fg='white')
lb2['bg'] = 'grey'
lb2.place(x=30, y=200)

text2 = Text(win, width=80, height=4, font=('Times New Roman', 12))
text2.place(x=30, y=230)

def openfile2():
    f = open("../Bộ dữ liệu/list2.txt", mode="r", encoding="utf-8")
    content = f.read()
    showContent(content)

s.configure("btnopen2.TButton",
            foreground="white",
            background="#FF9900",
            padding=(20, 5),
            font=('Times New Roman', 14),
            borderwidth=2,
            relief="solid",
            bordercolor ="#FF9966"
            )
btnopen2 = ttk.Button(win, text="Mở file dữ liệu thứ hai", command=openfile2, style="btnopen2.TButton")
btnopen2.place(x=720, y=230)
btnopen2.bind("<Enter>", on_enter)
def showContent(content):
    new_window = Tk()
    new_window.title('Nội dung tệp dữ liệu')
    new_window.geometry("+1150+170")
    text_widget = scrolledtext.ScrolledText(new_window, width=80, height=20, wrap=WORD)
    text_widget.insert(END, content)
    text_widget.pack(expand=True, fill='both')

    new_window.mainloop()

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

def phobertV2(text1,text2):
    # phobert-V2
    model_name = "vinai/phobert-base-v2"
    phobert = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids1 = tokenizer.encode(text1, return_tensors="pt")
    input_ids2 = tokenizer.encode(text2, return_tensors="pt")
    with torch.no_grad():
        outputs1 = phobert(input_ids1)
        outputs2 = phobert(input_ids2)
    vector1 = outputs1.last_hidden_state.mean(dim=1).squeeze().numpy()
    vector2 = outputs2.last_hidden_state.mean(dim=1).squeeze().numpy()
    return cosine_similarity([vector1], [vector2])[0][0]


def simCSE_Phobert(text1, text2):
    # SimCSE
    tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    tokens1 = tokenizer(tokenize(text1), return_tensors="pt")
    tokens2 = tokenizer(tokenize(text2), return_tensors="pt")

    # Tính toán embedding
    with torch.no_grad():
        embeddings1 = model(**tokens1).pooler_output
        embeddings2 = model(**tokens2).pooler_output

    # Tính toán độ tương đồng sử dụng cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

    return cosine_sim.item()

#bkai-foundation-model
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Phần tử đầu tiên của model_output chứa tất cả các token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def bkai(text1, text2):
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


def laydl():
    vb1 = text1.get("1.0", "end-1c").strip()
    vb2 = text2.get("1.0", "end-1c").strip()
    simcgia=""
    f = open("../Bộ dữ liệu/BoDuLieuCauDoCGDanhGia.txt", mode="r", encoding="utf-8")
    for s in f:
        s = s.replace('"', '')
        ss = s.split('\t')
        if (len(ss) != 5):
            print(s)
            continue
        if vb1 == ss[1] and vb2 == ss[2]:
             simcgia = "/"+str(int(ss[3]) / 4.0)
    f.close()
    return str(simcgia)
def click():
    vb1 = text1.get("1.0", "end-1c")
    vb2 = text2.get("1.0", "end-1c")
    simPhobertV2 = phobertV2(vb1,vb2)
    simCSE = simCSE_Phobert(vb1, vb2)
    simBkai= bkai(vb1,vb2)
    w2v=word2vec(vb1,vb2)
    simPhobertV2 = round(simPhobertV2, 4)
    simCSE = round(simCSE, 4)
    simBkai = round(simBkai, 4)
    w2v = round(w2v, 4)
    laydl()



    def update_progress(value):
        if vb1 != "" and vb2.strip() != "":  # So sánh ss[1] với text1
            bar['value'] = value
            if value < simPhobertV2 * 100:
                win.after(20, update_progress, value + 1)
                if simPhobertV2 < 0.6:
                    s.configure("phobert.Horizontal.TProgressbar", foreground='red', background='red')
                    lb5 = Label(win, text="PhoBert-V2: ", font=('Times New Roman', 14), bg="grey",
                                fg="white")
                    lb5.place(x=30, y=348)

                    lb6 = Label(win, text=" " + str(simPhobertV2)+laydl(), font=('Times New Roman', 14),
                                bg="grey",
                                fg="white")
                    lb6.place(x=490, y=349)
                    # win.after(4000, lambda: lb5.place_forget())
                elif simPhobertV2 >= 0.6 and simPhobertV2 < 0.8:
                    s.configure("phobert.Horizontal.TProgressbar", foreground='red', background='yellow')
                    lb5 = Label(win, text="PhoBert-V2: ", font=('Times New Roman', 14), bg="grey",
                                fg="white")
                    lb5.place(x=30, y=348)
                    lb6 = Label(win, text=" " + str(simPhobertV2)+laydl(), font=('Times New Roman', 14),
                                bg="grey",
                                fg="white")
                    lb6.place(x=490, y=349)
                else:
                    s.configure("phobert.Horizontal.TProgressbar", foreground='red', background='green')
                    lb5 = Label(win, text="PhoBert-V2: ", font=('Times New Roman', 14), bg="grey",
                                fg="white")
                    lb5.place(x=30, y=348)
                    lb6 = Label(win, text=" " + str(simPhobertV2)+laydl(), font=('Times New Roman', 14),
                                bg="grey",
                                fg="white")
                    lb6.place(x=490, y=349)

        else:
            messagebox.showwarning("Thông báo",
                                       "Văn bản thứ nhất hoặc thứ hai đang bị trống. Vui lòng điền đầy đủ trước khi so sánh")

    bar = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
                          style='phobert.Horizontal.TProgressbar')
    bar.place(x=180, y=350,height="25")
    update_progress(0)
    def update_progress2(value):
        if vb1 != "" and vb2.strip() != "":  # So sánh ss[1] với text1
            bar_CSE['value'] = value
            if value < simCSE * 100:
                win.after(20, update_progress2, value + 1)
                if simCSE <0.6:
                    s.configure("cse.Horizontal.TProgressbar", foreground='red', background='red')
                    lb5 = Label(win, text="SimCSE: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=388)

                    lb6 = Label(win, text=" " + str(simCSE)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=389)
                elif simCSE >=0.6 and simCSE <0.8:
                    s.configure("cse.Horizontal.TProgressbar", foreground='red', background='yellow')
                    lb5 = Label(win, text="SimCSE: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=388)

                    lb6 = Label(win, text=" " + str(simCSE)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=389)
                else:
                    s.configure("cse.Horizontal.TProgressbar", foreground='red', background='green')
                    lb5 = Label(win, text="SimCSE: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=388)

                    lb6 = Label(win, text=" " + str(simCSE)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=389)

    bar_CSE = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
                              style='cse.Horizontal.TProgressbar')
    bar_CSE.place(x=180, y=390,height="25")
    update_progress2(0)
    def update_progress3(value):
        if vb1 != "" and vb2.strip() != "":
            bar_bkai['value'] = value
            if value < simBkai * 100:
                win.after(20, update_progress3, value + 1)
                if simBkai <0.6:
                    s.configure("bkai.Horizontal.TProgressbar", foreground='red', background='red')
                    lb5 = Label(win, text="SimBkai: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=428)

                    lb6 = Label(win, text=" " + str(simBkai)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=429)
                elif simBkai >=0.6 and simBkai <0.8:
                    s.configure("bkai.Horizontal.TProgressbar", foreground='red', background='yellow')
                    lb5 = Label(win, text="SimBkai: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=428)

                    lb6 = Label(win, text=" " + str(simBkai)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=429)
                else:
                    s.configure("bkai.Horizontal.TProgressbar", foreground='red', background='green')
                    lb5 = Label(win, text="SimBkai: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=428)

                    lb6 = Label(win, text=" " + str(simBkai)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=429)

    bar_bkai = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
                              style='bkai.Horizontal.TProgressbar')
    bar_bkai.place(x=180, y=430,height="25")
    update_progress3(0)
    def update_progress4(value):
        if vb1 != "" and vb2.strip() != "":
            bar_w2v['value'] = value
            if value < w2v * 100:
                win.after(20, update_progress4, value + 1)
                if w2v <0.6:
                    s.configure("w2v.Horizontal.TProgressbar", foreground='red', background='red')
                    lb5 = Label(win, text="Word2Vec: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=468)

                    lb6 = Label(win, text=" " + str(w2v)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=469)
                elif w2v >=0.6 and w2v <0.8:
                    s.configure("w2v.Horizontal.TProgressbar", foreground='red', background='yellow')
                    lb5 = Label(win, text="Word2Vec: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=468)

                    lb6 = Label(win, text=" " + str(w2v)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=469)
                else:
                    s.configure("w2v.Horizontal.TProgressbar", foreground='red', background='green')
                    lb5 = Label(win, text="Word2Vec: ", font=('Times New Roman', 14), bg="grey",
                                    fg="white")
                    lb5.place(x=30, y=468)

                    lb6 = Label(win, text=" " + str(w2v)+laydl(), font=('Times New Roman', 14),
                                    bg="grey",
                                    fg="white")
                    lb6.place(x=490, y=469)

    bar_w2v = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
                              style='w2v.Horizontal.TProgressbar')
    bar_w2v.place(x=180, y=470,height="25")
    update_progress4(0)

def clear():
    text1.delete(1.0, END)
    text1.focus()
    text2.delete(1.0, END)
    clear_labels_and_progressbars()

s.configure("btnss.TButton",
            foreground="white",
            background="#00CC00",
            padding=(55, 5),
            font=('Times New Roman', 14),
            borderwidth=2,
            relief="solid",
            bordercolor ="#33CC66"
            )

btnss = ttk.Button(win, text="So Sánh", command=click, style="btnss.TButton")
btnss.place(x=720, y=340)
btnss.bind("<Enter>", on_enter)

s.configure("btnlm.TButton",
            foreground="white",
            background="#FF0000",
            padding=(55, 5),
            font=('Times New Roman', 14),
            borderwidth=2,
            relief="solid",
            bordercolor ="#FF3333"
            )
btnlm = ttk.Button(win, text="Làm Mới", command=clear,style="btnlm.TButton")
btnlm.bind("<Enter>", on_enter)
btnlm.place(x=720, y=450)

win.mainloop()

#Ông Nguyễn Bá Thụ, cục trưởng Cục Kiểm lâm VN, thừa nhận tình trạng vận chuyển, buôn bán hổ và xương hổ trái phép vẫn đang diễn ra tại VN và đặc biệt sôi động ở Hà Nội. Thật tuyệt vời, các khối hình đều giống nhau giống nhau một cách đáng kinh ngạc
#Ông Thụ, cục trưởng Cục Kiểm lâm VN, thừa nhận tình trạng vận chuyển, buôn bán hổ và xương hổ trái phép vẫn đang diễn ra tại VN và đặc biệt sôi động ở Hà Nội. Thật kinh ngạc, 42 khối vuông hình sắc cạnh không hơn kém nhau 1 gam

# Người Mã Liềng sinh sống ở vùng rừng giáp Lào, thuộc miền tây Hà Tĩnh, trải qua cuộc sống văn hóa độc đáo, nơi mà những đồng bào này sáng tạo những nhạc cụ như ống nứa và đàn trbon. Cuộc sống của họ đặc biệt, ẩn mình trên đỉnh núi Ka Đay, trong bản Rào Tre, nổi bật với những đặc điểm riêng biệt và tính kín đáo. Vào năm 1958, một đội chiến sĩ biên phòng phát hiện nhóm người Mã Liềng đang sống giấu kín trong rừng

# Người Mã Liềng ở vùng rừng giáp Lào thuộc miền tây Hà Tĩnh sống cuộc sống văn hóa độc đáo, nơi chỉ có ống nứa và đàn trbon làm nhạc cụ. Cuộc sống của họ, ẩn mình trên đỉnh núi Ka Đay, trong bản Rào Tre, đầy những nét độc đáo và kín đáo. Năm 1958, chiến sĩ biên phòng phát hiện nhóm người Mã Liềng sống chui rúc trong rừng. Đó là lúc ông Hồ Hiến bắt đầu "cắm bản" và ở lại hỗ trợ họ suốt 24 năm

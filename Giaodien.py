from tkinter import *
from tkinter import messagebox, ttk
from tkinter.ttk import Progressbar, Style
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer

win = Tk()

win.title('Phần Mềm So Sánh Văn Bản')
win.geometry('760x400')
win['bg'] = 'grey'
win.attributes('-topmost', True)

lb1 = Label(win, text="Văn Bản Thứ Nhất", font=('Times New Roman', 12), bg='grey', fg='white')
lb1.place(x=30, y=30)

lb2 = Label(win, text="Văn Bản Thứ Hai", font=('Times New Roman', 12), fg='white')
lb2['bg'] = 'grey'
lb2.place(x=400, y=30)

text1 = Text(win, width=40, height=3, font=('Times New Roman', 12))
text1.place(x=30, y=60)
text1.focus()

text2 = Text(win, width=40, height=3, font=('Times New Roman', 12))
text2.place(x=400, y=60)

model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_vector(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = phobert(input_ids)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector


def click():

    vb1 = text1.get("1.0", "end-1c")
    vb2 = text2.get("1.0", "end-1c")

    vector1 = get_vector(vb1)
    vector2 = get_vector(vb2)
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    s = ttk.Style()
    s.theme_use('clam')

    def update_progress(value):
        if vb1 != "" and vb2.strip() != "":  # So sánh ss[1] với text1
            bar['value'] = value
            if value < similarity * 100:
                win.after(20, update_progress, value + 1)
                if similarity < 0.6:
                    s.configure("Horizontal.TProgressbar", foreground='red', background='red')
                    lb5 = Label(win, text="Cosine similarity: ", font=('Times New Roman', 14), bg="grey",
                                fg="white")
                    lb5.place(x=30, y=296)

                    lb6 = Label(win, text=" " + str(similarity), font=('Times New Roman', 14),
                                bg="grey",
                                fg="red")
                    lb6.place(x=490, y=296)
                    # win.after(4000, lambda: lb5.place_forget())
                elif similarity >= 0.6 and similarity < 0.8:
                    s.configure("Horizontal.TProgressbar", foreground='red', background='yellow')
                    lb5 = Label(win, text="Cosine similarity: ", font=('Times New Roman', 14), bg="grey",
                                fg="white")
                    lb5.place(x=30, y=296)
                    lb6 = Label(win, text=" " + str(similarity), font=('Times New Roman', 14),
                                bg="grey",
                                fg="yellow")
                    lb6.place(x=490, y=296)
                    # win.after(4000, lambda: lb5.place_forget())
                else:
                    s.configure("Horizontal.TProgressbar", foreground='red', background='green')
                    lb5 = Label(win, text="Cosine similarity: ", font=('Times New Roman', 14),bg="grey",
                                fg="white")
                    lb5.place(x=30, y=296)
                    lb6 = Label(win,text=" "+ str(similarity), font=('Times New Roman', 14),
                                bg="grey",
                                fg="green")
                    lb6.place(x=490, y=296)
                    # win.after(4000, lambda: lb5.place_forget())
        else:
            messagebox.showwarning("Thông báo",
                                   "Văn bản thứ nhất hoặc thứ hai đang bị trống. Vui lòng điền đầy đủ trước khi so sánh")

    bar = ttk.Progressbar(win, orient="horizontal", length=300, mode='determinate',
                          style='Horizontal.TProgressbar')
    bar.place(x=180, y=300)
    update_progress(0)

def clear():
    text1.delete(1.0, "end")
    text1.focus()
    text2.delete(1.0, "end")



btnss = Button(win, text="So Sánh", font=('Times New Roman', 14), bg='yellow', fg='black', width=10, height=2,
               command=click)
btnss.place(x=330, y=170)

btnss = Button(win, text="Clear", font=('Times New Roman', 14), bg='yellow', fg='black', width=10, height=2,
               command=clear)
btnss.place(x=330, y=230)


win.mainloop()

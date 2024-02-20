from tkinter import *
from tkinter import messagebox

from pexpect.ANSI import term

win = Tk() #khai báo biến
win.title('Phần Mềm So Sánh Văn Bản') # đặt tiêu đề cho tiêu đề
win.geometry('760x400') # đặt kích thước cho giao diện
win['bg'] = 'grey' #thay đổi background
win.attributes('-topmost',True) #Giao diện luôn được ưu tiên

#Tạo label
lb1=Label(win,text="Văn Bản Thứ Nhất",font=('Times New Roman',12),bg='grey',fg='white')
lb1.place(x=30,y=30) # x là chiều ngang , y là chiều dọc
lb2=Label(win,text="Văn Bản Thứ Hai",font=('Times New Roman',12),fg='white')
lb2['bg']='grey'
lb2.place(x=400,y=30) # x là chiều ngang , y là chiều dọc

#Tạo Text
text1=Text(win,width=40,height=3, font=('Times New Roman',12))
text1.place(x=30,y=60)
text1.focus()

text2=Text(win,width=40,height=3, font=('Times New Roman',12))
text2.place(x=400,y=60)

def text1_enter(event):
    f = open("data1.txt", mode="r", encoding="utf-8")
    for s in f:
        s = s.replace('"', '')
        ss = s.split('\t')
        if ss[1] == text1.get("1.0", "end-1c").strip():  # So sánh ss[1] với text1
            text2.insert("1.0", ss[2])
    f.close()

text1.bind("<Return>", text1_enter)  #bắt sự kiện enter cho text1

#Tạo button
def click():
    # messagebox.showinfo("Thông báo", "So sánh thành công!")
    lb3 = Label(win, text="Văn bản 1: "+text1.get("1.0", "end-1c"), font=('Times New Roman', 12), bg='red', fg='white')
    lb3.place(x=30, y=270)
    win.after(4000, lambda: lb3.place_forget()) #lb3 sau 2s sẽ biến mất
    lb4 = Label(win, text="Văn bản 2: " + text2.get("1.0", "end-1c"), font=('Times New Roman', 12), bg='red',fg='white')
    lb4.place(x=30, y=300)
    win.after(4000, lambda: lb4.place_forget())
    f = open("data1.txt", mode="r", encoding="utf-8")
    for s in f:
        s = s.replace('"', '')
        ss = s.split('\t')
        if text1.get("1.0", "end-1c").strip() != "" and text2.get("1.0","end-1c").strip() != "":  # So sánh ss[1] với text1
            if ss[1] == text1.get("1.0", "end-1c").strip():
                temp = float(ss[3])
                if temp < 0.6:
                    lb5 = Label(win, text="Cosine similarity: " + str(temp), font=('Times New Roman', 14), bg='cyan',
                                fg='red')
                    lb5.place(x=30, y=350)
                    win.after(4000, lambda: lb5.place_forget())
                elif temp >= 0.6 and temp < 0.8:
                    lb5 = Label(win, text="Cosine similarity: " + str(temp), font=('Times New Roman', 14), bg='cyan',
                                fg='#FFFF66')
                    lb5.place(x=30, y=350)
                    win.after(4000, lambda: lb5.place_forget())
                else:
                    lb5 = Label(win, text="Cosine similarity: " + str(temp), font=('Times New Roman', 14), bg='cyan',
                                fg="green")
                    lb5.place(x=30, y=350)
                    win.after(4000, lambda: lb5.place_forget())
        else:
            messagebox.showwarning("Thông báo",
                                   "Văn bản thứ nhất hoặc thứ hai đang bị trống. Vui lòng điền đầy đủ trước khi so sánh")
            break
    f.close()

btnss = Button(win,text="So Sánh", font=('Times New Roman',14),bg='yellow',fg='black', width=10, height=2, command = click) # khi ấn vào sẽ thực hiện hàm click

btnss.place(x=330,y=170)

win.mainloop() #giúp hiện cửa sổ liên tục
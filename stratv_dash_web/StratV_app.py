from tkinter import *

import os
import webbrowser

app = Tk()

def visual():
    from subprocess import Popen

    Popen('python StratV.py')
    url = "http://localhost:8050/"
    webbrowser.open_new(url)

app.iconbitmap('favicon.ico')
app.title('Strat V')
app.resizable(0, 0)

welcome_message = Label(app, text='WELCOME TO THE WATERFLOOD METHOD SOFTWARE',font=15)
welcome_message.place(rely=0.05, relx=0.1)

method_frame = LabelFrame(app, text='Methods')
method_frame.place(relheight=0.4,relwidth=0.5,rely=0.2, relx=0.1)

Dykstra_label = Label(method_frame, text='- Dykstra-Parsons')
Dykstra_label.place(rely=0.05, relx=0.05)
Reznik_label = Label(method_frame, text='- Reznik et al')
Reznik_label.place(rely=0.3, relx=0.05)
Robert_label = Label(method_frame, text='- Robert')
Robert_label.place(rely=0.55, relx=0.05)

run_program_button = Button(app, text='Run Program',cursor='hand2',fg='white',bg='green', command= visual)
run_program_button.place(relwidth=0.2, rely=0.7, relx=0.1)

Copywrite = Label(app, text='© 2020, AUST', fg='white', bg='black').place(rely=0.95, relx=0.75)
app.geometry("500x400")
app.mainloop()
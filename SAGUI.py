import SentimentAnalysis
import torch
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


class SAGUI():
    def __init__(self):
        root = tk.Tk()
        root.geometry('900x700')
        root.title('Sentiment Analysis')

        intro_frame = tk.Frame(root)
        create_model_frame = tk.Frame(root)

        for f in (intro_frame, create_model_frame):
            f.grid(row=0,column=0,sticky=tk.N + tk.E + tk.S + tk.W)

        #intro frame
        title_label = tk.Label(intro_frame, text='Sentiment Analysis', font=('Arial', 20))
        title_label.pack(padx=20, pady=10)

        button_frame = tk.Frame(intro_frame)

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        create_model_button = tk.Button(button_frame, text="Create new model", font = ('Arial', 16), width=20, command=lambda: create_model_frame.tkraise())
        create_model_button.grid(row=0, column=0, padx=30, sticky=tk.E+tk.W)

        load_model_button = tk.Button(button_frame, text="Load model", font = ('Arial', 16), width=20, command = lambda: messagebox.showerror(title="Error", message="Not implemented"))
        load_model_button.grid(row=0, column=1, padx=30, sticky=tk.E+tk.W)

        button_frame.pack(pady=50)

        #model frame
        exit_create_model_button = tk.Button(create_model_frame, text='Back', font = ('Arial', 16), width=20, command=lambda: intro_frame.tkraise())
        exit_create_model_button.pack()

        intro_frame.tkraise()

        root.mainloop()

SAGUI()
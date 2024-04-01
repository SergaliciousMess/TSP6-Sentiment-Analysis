import SentimentAnalysis
import torch
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.geometry('900x700')
root.title('Sentiment Analysis')

title_label = tk.Label(root, text='Sentiment Analysis', font=('Arial', 20))
title_label.pack(padx=20, pady=10)

button_frame = tk.Frame(root)

button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)

create_model_button = tk.Button(button_frame, text="Create new model", font = ('Arial', 16), width=20)
create_model_button.grid(row=0, column=0, padx=30, sticky=tk.E+tk.W)

load_model_button = tk.Button(button_frame, text="Load model", font = ('Arial', 16), width=20)
load_model_button.grid(row=0, column=1, padx=30, sticky=tk.E+tk.W)

button_frame.pack(pady=50)

root.mainloop()
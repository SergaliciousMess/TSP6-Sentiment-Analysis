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

        container = tk.Frame(root)
        container.pack()

        self.intro_frame = tk.Frame(container)
        self.create_model_frame = tk.Frame(container)
        self.model_loaded_frame = tk.Frame(container)
        self.train_model_frame = tk.Frame(container)
        self.evaluate_model_frame = tk.Frame(container)

        for f in (self.intro_frame, self.create_model_frame, self.model_loaded_frame, self.train_model_frame, self.evaluate_model_frame):
            f.grid(row=0,column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        

        #intro frame
        title_label = tk.Label(self.intro_frame, text='Sentiment Analysis', font=('Arial', 20))
        title_label.pack(padx=20, pady=10)

        intro_button_frame = tk.Frame(self.intro_frame)

        intro_button_frame.columnconfigure(0, weight=1)
        intro_button_frame.columnconfigure(1, weight=1)

        create_model_button = tk.Button(intro_button_frame, text="Create new model", font = ('Arial', 16), width=20, command = lambda: self.create_model_frame.tkraise())
        create_model_button.grid(row=0, column=0, padx=30, sticky=tk.E+tk.W)

        load_model_button = tk.Button(intro_button_frame, text="Load model", font = ('Arial', 16), width=20, command = lambda: messagebox.showerror(title="Error", message="Not implemented"))
        load_model_button.grid(row=0, column=1, padx=30, sticky=tk.E+tk.W)

        intro_button_frame.pack(pady=50)

        #create model frame
        exit_create_model_button = tk.Button(self.create_model_frame, text='Back', font = ('Arial', 16), width=20, command = lambda: self.intro_frame.tkraise())
        exit_create_model_button.pack(pady=30)

        parameters = tk.Frame(self.create_model_frame)

        data_entry_label = tk.Label(parameters, text="Dataset:", font=('Arial', 12))
        data_entry_label.grid(row=0, column=0)

        self.data_entry = tk.Entry(parameters, text="dataset")
        self.data_entry.insert(0, "test_data.csv")
        self.data_entry.grid(row=0,column=1)

        embedding_dim_label = tk.Label(parameters, text="Embedding Dimension:", font=('Arial', 12))
        embedding_dim_label.grid(row=1,column=0)

        self.embedding_dim_entry = tk.Entry(parameters, text="embedding_dim")
        self.embedding_dim_entry.insert(0, "64")
        self.embedding_dim_entry.grid(row=1,column=1)

        batch_size_label = tk.Label(parameters, text="Batch Size:", font=('Arial', 12))
        batch_size_label.grid(row=2,column=0)

        self.batch_size_entry = tk.Entry(parameters, text="batch_size")
        self.batch_size_entry.insert(0, "512")
        self.batch_size_entry.grid(row=2,column=1)

        learning_rate_label = tk.Label(parameters, text="Learning Rate:", font=('Arial', 12))
        learning_rate_label.grid(row=3,column=0)

        self.learning_rate_entry = tk.Entry(parameters, text="learning_rate")
        self.learning_rate_entry.insert(0, "1")
        self.learning_rate_entry.grid(row=3,column=1)

        parameters.pack()

        create_model_button = tk.Button(self.create_model_frame, text='Create Model', font = ('Arial', 16), width=20, command = self.create_model)
        create_model_button.pack(pady=30)

        #model loaded frame
        model_loaded_label = tk.Label(self.model_loaded_frame, text="Model loaded", font=('Arial', 20))
        model_loaded_label.pack()

        loaded_button_frame = tk.Frame(self.model_loaded_frame)
        loaded_button_frame.columnconfigure(0, weight=1)
        loaded_button_frame.columnconfigure(1, weight=1)

        evaluate_button = tk.Button(loaded_button_frame, text="Evaluate", font = ('Arial', 16), width=20, command = lambda: self.evaluate_model_frame.tkraise())
        evaluate_button.grid(row=0,column=0,padx=20,pady=10)

        train_model_button = tk.Button(loaded_button_frame, text="Train", font = ('Arial', 16), width=20, command = lambda: self.train_model_frame.tkraise())
        train_model_button.grid(row=0,column=1,padx=20,pady=10)

        unload_model_button = tk.Button(loaded_button_frame, text="Back", font = ('Arial', 16), width=20, command = self.clear_model)
        unload_model_button.grid(row=1,column=0,padx=20,pady=10)

        save_button = tk.Button(loaded_button_frame, text="Save", font = ('Arial', 16), width=20, command = lambda: messagebox.showerror(title="Error", message="Not implemented"))
        save_button.grid(row=1,column=1,padx=20,pady=10)

        loaded_button_frame.pack()

        #train model frame
        exit_training_button = tk.Button(self.train_model_frame, text='Back', font = ('Arial', 16), width=20, command = lambda: self.model_loaded_frame.tkraise())
        exit_training_button.pack(pady=30)

        train_model_input_frame = tk.Frame(self.train_model_frame)
        train_model_input_frame.columnconfigure(0, weight=1)
        train_model_input_frame.columnconfigure(1, weight=1)

        epochs_label = tk.Label(train_model_input_frame, text="Epochs:", font=('Arial', 20))
        epochs_label.grid(row=0,column=0)

        self.epochs_entry = tk.Entry(train_model_input_frame, text="epochs")
        self.epochs_entry.insert(0, "2000")
        self.epochs_entry.grid(row=0,column=1)

        train_model_input_frame.pack()

        start_training_button = tk.Button(self.train_model_frame, text ="Train", font = ('Arial', 16), width=20, command = self.train)
        start_training_button.pack(pady=30)

        #evaluate model frame
        exit_evaluation_button = tk.Button(self.evaluate_model_frame, text='Back', font = ('Arial', 16), width=20, command = lambda: self.model_loaded_frame.tkraise())
        exit_evaluation_button.pack(pady=30)

        self.analyze_textbox = tk.Text(self.evaluate_model_frame)
        self.analyze_textbox.pack()

        analyze_button = tk.Button(self.evaluate_model_frame, text='Analyze', font = ('Arial', 16), width=20, command = self.analyze_message)
        analyze_button.pack(pady=30)

        self.prediction_label = tk.Label(self.evaluate_model_frame, text='Prediction: ', font = ('Arial', 16))
        self.prediction_label.pack()

        self.intro_frame.tkraise()
        root.mainloop()

    def create_model(self):
        try:
            dataset = SentimentAnalysis.load_dataset(self.data_entry.get(), "csv")
        except:
            messagebox.showerror(title="Invalid argument", message="Dataset could not be loaded. Name of file might be wrong.")
            return

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            batch_size = int(self.batch_size_entry.get())
        except:
            messagebox.showerror(title="Invalid argument", message="Error in batch size argument")
            return
        if batch_size <= 0:
            messagebox.showerror(title="Invalid argument", message="Batch size must be greater than 0.")
            return
        
        dataloader = DataLoader(dataset, generator=torch.Generator(device=device), batch_size=batch_size)
        self.dataloader = dataloader

        try:
            lr = float(self.learning_rate_entry.get())
        except:
            messagebox.showerror(title="Invalid argument", message="Error in learning rate argument")
            return
        if lr <= 0:
            messagebox.showerror(title="Invalid argument", message="Learning rate must be greater than 0.")
            return
        
        try:
            embedding_dim = int(self.embedding_dim_entry.get())
        except:
            messagebox.showerror(title="Invalid argument", message="Error in embedding dimension argument")
            return
        if embedding_dim <= 0:
            messagebox.showerror(title="Invalid argument", message="Embedding dimension must be greater than 0.")
            return
        
        try:
            self.model = SentimentAnalysis.SentimentAnalysis(dataloader, embedding_width=embedding_dim, learning_rate=lr, device=device)
            self.model_loaded_frame.tkraise()
        except:
            messagebox.showerror(title="Error", message="Model could not be created.")

    def clear_model(self):
        self.model = None
        self.dataloader = None
        self.intro_frame.tkraise()

    def analyze_message(self):
        prediction = self.model.analyze([self.analyze_textbox.get("1.0", "end-1c")])[0]
        self.prediction_label.config(text="Prediction: " + prediction)

    def train(self):
        e = int(self.epochs_entry.get())
        if e <= 0:
            messagebox.showerror(title="Invalid argument", message="Epochs must be greater than zero.")
            return
        
        if e >= 3000:
            response = messagebox.askquestion("Warning", "Large values for epochs may cause program to become temporarily unresponsive. Do you wish to continue?")
            if response == "no":
                return
            
        
        self.model.train_from_dataloader(self.dataloader, epochs=e)
        messagebox.showinfo("Training complete","Training complete")

SAGUI()
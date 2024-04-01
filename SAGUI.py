import SentimentAnalysis
import torch
from torch.utils.data import DataLoader


data = SentimentAnalysis.load_dataset('test_data.csv', 'csv')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataloader = DataLoader(data, batch_size = 128, shuffle=True, generator=torch.Generator(device=device))

model = SentimentAnalysis.SentimentAnalysis(dataloader=dataloader, device=device)

print("starting training: " + device)
model.train_from_dataloader(dataloader=dataloader, epochs=2)
print(model.analyze(['I love life!','I hate life!','monday, am i right',"weather is looking pretty good today"]))
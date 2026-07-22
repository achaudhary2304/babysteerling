import os
import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
folder = "./data/tinyshakespeare/"
filename = "input.txt"

os.makedirs(folder, exist_ok=True)
path = os.path.join(folder, filename)
urllib.request.urlretrieve(url, path)
print(f"Downloaded {filename} successfully.")

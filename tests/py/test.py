import requests

url = 'http://127.0.0.1:8000/imagezap'
data = {"images" : ["https://static.wixstatic.com/media/ba2cd3_501e0b17405a4747bd94ed06ec55799f~mv2.png", "https://static.wixstatic.com/media/ba2cd3_24355b959d064e39a3da7b4f322f91f7~mv2.jpg"]}
resp = requests.post(url=url, json=data) 
print(resp.json())
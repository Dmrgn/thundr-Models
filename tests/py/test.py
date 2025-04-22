import requests

url = 'http://127.0.0.1:8000/imagezap'
files = [('files', open('data/test1.png', 'rb')), ('files', open('data/test2.png', 'rb'))]
resp = requests.post(url=url, files=files) 
print(resp.json())
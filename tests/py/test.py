import requests

url = 'http://127.0.0.1:8000/imagezap'
data = {"images" : ["https://thundr.ca/thundr.png", "https://thundr.ca/_nuxt/laptop1.BFB2ld_T.png"]}
resp = requests.post(url=url, json=data) 
print(resp.json())
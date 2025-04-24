import requests

url = 'http://127.0.0.1:8000/imagezap'
data = {"images" : ["https://static.wixstatic.com/media/ba2cd3_501e0b17405a4747bd94ed06ec55799f~mv2.png/v1/fill/w_317,h_92,al_c,q_85,enc_avif,quality_auto/ba2cd3_501e0b17405a4747bd94ed06ec55799f~mv2.png"]}
resp = requests.post(url=url, json=data) 
print(resp.json())
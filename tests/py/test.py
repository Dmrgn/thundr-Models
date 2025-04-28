import requests
import numpy as np

url = 'http://127.0.0.1:8000/imagezap'
data = {"images" : [
    "https://aplconstruction.ca/wp-content/uploads/2020/08/future-building-construction-engineering-project-1200x1200.png",
    # "https://aplconstruction.ca/wp-content/uploads/2023/05/APL_Home_Plans-150x150.webp",
    "https://aplconstruction.ca/wp-content/uploads/2020/08/future-building-construction-engineering-project.png",
    # "https://aplconstruction.ca/wp-content/uploads/2023/05/APL_Home_Plans-300x300.webp",
    # "https://aplconstruction.ca/wp-content/uploads/2020/08/333.jpg",
]}
resp = requests.post(url=url, json=data).json()

print(resp)

one = np.array([resp[0][key] for key in resp[0].keys()])
two = np.array([resp[1][key] for key in resp[1].keys()])
print(np.linalg.norm(one - two))

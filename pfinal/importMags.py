import pandas as pd
import numpy as np
import requests
import shutil

df = pd.read_csv('./magazines.csv')
for row in range(len(df)):
  for page in range(1, 200):
    url = df["Magazine url"][row]+'/files/assets/common/page-html5-substrates/page'+str(page).zfill(4)+'.jpg'
    r = requests.get(url, stream=True)
    print (r.raw)
    print (row, page, url, r.status_code)
    if (r.status_code==requests.codes.ok):
      r.raw.decode_content = True
      with open("./data/mags/{}-{}.jpg".format(df["Id"][row], page), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
          f.write(chunk)
    else:
      break

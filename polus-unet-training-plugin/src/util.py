import zipfile
import requests
from pathlib import Path

PATH = Path("caffemodels.zip")
URL = "https://lmb.informatik.uni-freiburg.de/lmbsoft/unet/caffemodels.zip"
content = requests.get(URL).content
PATH.open("wb").write(content)
with zipfile.ZipFile("caffemodels.zip","r") as zip_ref:
    zip_ref.extractall(".")
from urllib.request import urlopen
from random import shuffle

selection = r"Z:\Jeffrey-Ede\crystal_structures\cod-inorganic\COD-selection.txt"
save_loc = r"Z:\Jeffrey-Ede\crystal_structures\cod-inorganic\inorganic_cifs\\"
num_to_download = 10_000


with open(selection, "r") as f:
    urls = f.read()
    urls = urls.split("\n")
    urls = urls[:-1]

    shuffle(urls)

    num_downloaded = 0
    for url in urls:

        if num_downloaded >= num_to_download:
            break

        download = urlopen(url).read()
        with open(f"{save_loc}{num_downloaded}.cif", "wb") as w:
            w.write(download)

        num_downloaded += 1

        try:
            download = urlopen(url).read()
            with open(f"{save_loc}{num_downloaded}.cif", "wb") as w:
                w.write(download)

            num_downloaded += 1
        except:
            pass
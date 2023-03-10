import requests
import re
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True,
				help='path to the image')

args = ap.parse_args()

filename = args.image
files = {'my_file': (filename, open(filename, 'rb'))}
json = {'programmNr': "Hello", 'second': "World"}

response = requests.post(
    # 'http://192.168.8.113:8000/file',
	'http://127.0.0.1:8000/file',
    files=files,
    data={'programmNr': "Hello", 'second': "World"}
)
path = 'precon_web_folder'

if response.status_code == 200:
	print('Yeah')
	d = response.headers['content-disposition']
	fname = re.findall("filename=(.+)", d)[0]
	fname = fname.replace('"', '')
	with open(f"{path}/{fname}", 'wb') as f:
		response.raw.decode_content = True
		f.write(response.content)

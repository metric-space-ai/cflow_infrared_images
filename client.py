import requests
import re
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True,
				help='path to the image')

args = ap.parse_args()

filename = args.image
files = {'my_file': (filename, open(filename, 'rb'))}
json = {'first': "Hello", 'second': "World"}

response = requests.post(
    'http://192.168.8.113:8000/file',
	# 'http://127.0.0.1:8000/file',
    files=files,
    data={'first': "Hello", 'second': "World"}
)
path = '/home/nvidia/Downloads'

if response.status_code == 200:
	print('Yeah')
	d = response.headers['content-disposition']
	fname = re.findall("filename=(.+)", d)[0]
	fname = fname.replace('"', '')
	with open(f"{path}/{fname}", 'wb') as f:
		response.raw.decode_content = True
		f.write(response.content)

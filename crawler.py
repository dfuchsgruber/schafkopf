
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import os
from getpass import getpass

username = 'domi.erdnuss@gmx.de'
password = getpass(f'Password for email "{username}": ')

s = requests.Session()

# Get a new auth token for this session
resp = s.get('https://www.sauspiel.de/')
soup = BeautifulSoup(resp.content)
a_token = soup.select_one("input[name='authenticity_token']")['value']

# Perform login for this auth token
login_data =  {'login': username, 'password':password, 'authenticity_token':a_token, 'remember_me':'1', 'utf8':'✓'}
r = s.post('https://www.sauspiel.de/login', data=login_data)

# Binary search to find first non existent game
os.listdir('games')

for i in tqdm(range(100000)):
    if i < 30463:
        continue
    game_idx = 1000000000 + i
    r = s.get(f'https://www.sauspiel.de/spiele/{game_idx}.json')
    if not os.path.exists(f'games/game_{game_idx}.json'):
        with open(f'games/game_{game_idx}.json', 'wb') as f:
            f.write(r.content)




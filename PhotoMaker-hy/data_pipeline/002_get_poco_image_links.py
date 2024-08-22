# 说明：爬poco前，先要爬work id

import os
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import argparse
# import chromedriver_autoinstaller
from selenium.webdriver.chrome.options import Options
import csv
from webdriver_manager.chrome import ChromeDriverManager


SCROLL_PAUSE_TIME = 10

parser = argparse.ArgumentParser(description='pexels get meta information')
parser.add_argument('--save_root', type=str, help='The save root')
parser.add_argument('--save_meta_name', type=str, help='The save meta name')
opt = parser.parse_args()

# chromedriver_autoinstaller.install()
options = Options()
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
referer_url = f"https://www.poco.cn/works/works_list?classify_type=1&works_type=editor/"
driver.get(f"{referer_url}")
save_root = opt.save_root
save_meta_name = opt.save_meta_name

os.makedirs(f'{save_root}', exist_ok=True)

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

print(last_height)

recorded_ids = []
save_first_step_link_txt = f'{save_root}/{save_meta_name}'

if os.path.exists(save_first_step_link_txt):
    with open(save_first_step_link_txt, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        data = list(reader)
        for line in data[1:]:
            recorded_ids.append(line[0])
# write tsv
with open(f'{save_first_step_link_txt}', 'a+', newline='') as f:
    tsv_w = csv.writer(f, delimiter='\t')

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        soup = BeautifulSoup(driver.page_source)
        # print(len())
        # exit()
        for links in soup.find_all('a', {'class': 'vw_works_part'}):
            # img_link = links.find('img').get('src').split('?')[0]
            work_link = links.get('href')
            work_id = work_link.split('/')[-1].split('_')[-1]
            if work_id in recorded_ids:
                print('skip', work_link)
                continue
                
            string = f"{work_id}\t{work_link}"
            tsv_w.writerow([work_id, work_link])
            recorded_ids.append(work_link)
            print(len(recorded_ids), string)

        ### TODO:
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        # if new_height == last_height:
        #     break
        last_height = new_height
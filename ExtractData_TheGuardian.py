#################################################################
##  Script Info: It extracts the news from TheGuardian API 
##  Author: Mohammed Habibllah Baig 
##  Date : 11/22/2017
#################################################################

import json
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

TEMP_DIR = join('tempdata', 'articles')
makedirs(TEMP_DIR, exist_ok=True)

###################################################################
################## SETTING URL PARAMS #############################
###################################################################

API_KEY = "REMOVED"
API_ENDPOINT = 'http://content.guardianapis.com/search'
API_params = {
    'from-date': "",
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': API_KEY
}

#####################################################################
######### CODE TO EXTRACT THE NEWS ARTICLES FOR EACH DATE ###########
#####################################################################

start_date = date(2016, 8, 1)
end_date = date(2017,1, 15)
date_range = range((end_date - start_date).days + 1)

#Loop for dates
for day in date_range:
    dt = start_date + timedelta(days=day)
    date_str = dt.strftime('%Y-%m-%d')
    fileName = join(TEMP_DIR, date_str + '.json')
    if not exists(fileName):
        results = []
        API_params['from-date'] = date_str
        API_params['to-date'] = date_str
        cur_page = 1
        total_pages = 1
        # Loop for all the Pages
        while cur_page <= total_pages:
            API_params['page'] = cur_page
            resp = requests.get(API_ENDPOINT, API_params)
            data = resp.json()
            results.extend(data['response']['results'])
            # if there is more than one page
            cur_page += 1
            total_pages = data['response']['pages']

        with open(fileName, 'w') as f:
            f.write(json.dumps(results, indent=2))

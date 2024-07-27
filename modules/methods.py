import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
from abc import ABC, abstractmethod

import login_info
from modules.DataProcessor import DataProcessor

class Scraper(DataProcessor):
    def __init__(self):
        super().__init__()
        self._login()

    def _login(self):
        self.login_info = {
            "login_id": login_info.USERID,
            "pswd": login_info.PASSWORD,
        }
        self.session = requests.session()
        self.url_login = "https://regist.netkeiba.com/account/?pid=login&action=auth"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # New login process
        login_page = self.session.get(self.url_login, headers=self.headers)
        soup = bs(login_page.content, "html.parser")
        for hidden_input in soup.find_all("input", type="hidden"):
            self.login_info[hidden_input["name"]] = hidden_input["value"]

        self.ses = self.session.post(
            self.url_login,
            data=self.login_info,
            headers=self.headers,
            allow_redirects=False,
        )

        if self.ses.status_code == 302:
            print("session 302")
            redirect_url = self.ses.headers["Location"]
            self.ses = self.session.get(redirect_url, headers=self.headers)


        # print(f"Cookies: {self.session.cookies}")

    @abstractmethod
    def scrape(id_list: list):
        pass


class Preprocessor(DataProcessor):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def fill_null_values(columns: list):
        pass
    
    @abstractmethod
    def get_categorical_values(columns: list):
        pass

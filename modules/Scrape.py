import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm
from io import StringIO
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from modules.methods import Scraper


class RaceScraper(Scraper):
    def __init__(self):
        super().__init__()

        self.df_race = pd.DataFrame()
        self.df_odds = pd.DataFrame()
        self.race_results = {}
        self.odds_results = {}
        self.location_map = {
            "01": "札幌",
            "02": "函館",
            "03": "福島",
            "04": "新潟",
            "05": "東京",
            "06": "中山",
            "07": "中京",
            "08": "京都",
            "09": "阪神",
            "10": "小倉",
        }

    def scrape(self, id_list: list) -> pd.DataFrame:
        self.df_race = pd.DataFrame()
        self.df_odds = pd.DataFrame()
        self.race_results = {}
        self.odds_results = {}
        for race_id_no_round in tqdm(id_list):
            for i in range(1, 13):
                race_id = race_id_no_round + str(i).zfill(2)
                time.sleep(1)
                print(f"Scraping race {race_id}")
                try:
                    url = "https://db.sp.netkeiba.com/race/" + race_id
                    self.html = self.session.get(url, cookies=self.session.cookies)
                    self.html.encoding = "EUC-JP"
                    self.soup = BeautifulSoup(self.html.content, "html.parser")

                    self._get_race_basic_info()
                    # print("Got basic info")
                    self._get_horse_jockey_id()
                    # print("Got horse and jockey id")
                    self._get_pace()
                    # print("Got lap time")
                    self._get_odds()
                    self._set_index(race_id)

                # 存在しないrace_idを飛ばす
                except IndexError:
                    print(f"IndexErroe: Race {url} not found")
                    break
                except (
                    AttributeError
                ):  # 存在しないrace_idでAttributeErrorになるページもあるので追加
                    print(f"AttributeErroe: Race {url} not found")
                    break
                # wifiの接続が切れた時などでも途中までのデータを返せるようにする
                except Exception as e:
                    year = race_id_no_round[:4]
                    location_code = race_id_no_round[4:6]
                    location = self.location_map[location_code]
                    round_no = race_id_no_round[6:8]
                    day_no = race_id_no_round[8:10]
                    print(e, f"{year}年 {location} 第{round_no}回 {day_no}日目のレースは存在しません")
                    break

        # pd.DataFrame型にして一つのデータにまとめる
        if self.race_results:
            race_results_df = pd.concat([self.race_results[key] for key in self.race_results])
        else:
            race_results_df = pd.DataFrame()
        
        if self.odds_results:
            odds_results_df = pd.concat([self.odds_results[key] for key in self.odds_results])
        else:
            odds_results_df = pd.DataFrame()

        return race_results_df, odds_results_df

    def _get_race_basic_info(self):
        self.df_race = pd.read_html(StringIO(self.html.text))[0]
        self.df_race = self.df_race.rename(columns=lambda x: x.replace(" ", ""))  # Remove spaces in column names

        # Scrape weather, race type, course length, ground state, and date
        texts = (
                    self.soup.find("div", attrs={"class": "RaceHeader_Value"}).text
                    + self.soup.find("div", attrs={"class": "RaceHeader_Value_Others"}).text
                )

        date = self.soup.find("span", attrs={"class": "Race_Date"}).text.strip()
        self.df_race["date"] = [date] * len(self.df_race)

        info = re.findall(r"\w+", texts)
        for text in info:
            if "芝" in text:
                self.df_race["race_type"] = ["芝"] * len(self.df_race)
            if "ダ" in text:
                self.df_race["race_type"] = ["ダート"] * len(self.df_race)
            if "障" in text:
                self.df_race["race_type"] = ["障害"] * len(self.df_race)
            if "m" in text:
                self.df_race["course_len"] = [int(re.findall(r"\d+", text)[-1])] * len(self.df_race)
            if text in ["良", "稍", "重", "不"]:
                self.df_race["ground_state"] = [text] * len(self.df_race)
                if "障" in text:
                    self.df_race["ground_index"] = [0] * len(self.df_race)
                else:
                    baba_index = pd.read_html(StringIO(self.html.text))[2].iloc[0, 1]
                    self.df_race["ground_index"] = [baba_index] * len(self.df_race)
            if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                self.df_race["weather"] = [text] * len(self.df_race)
            if "年" in text:
                self.df_race["date"] = [text] * len(self.df_race)

    def _get_horse_jockey_id(self):
        # 馬ID、騎手IDをスクレイピング
        horse_id_list = []
        horse_a_list = (
            self.soup.find("table", attrs={"class": "table_slide_body ResultsByRaceDetail"})
            .find_all("tbody")[0]
            .find_all("a", attrs={"href": re.compile(r"horse/(\d+)/")})
        )
        for a in horse_a_list:
            horse_id = re.findall(r"\d+", a["href"])
            horse_id_list.append(horse_id[0])

        jockey_id_list = []
        jockey_a_list = (
            self.soup.find("table", attrs={"class": "table_slide_body ResultsByRaceDetail"})
            .find_all("tbody")[0]
            .find_all("a", attrs={"href": re.compile(r"jockey/(\d+)/")})
        )
        for a in jockey_a_list:
            jockey_id = re.findall(r"\d+", a["href"])
            jockey_id_list.append(jockey_id[0])

        self.df_race["horse_id"] = horse_id_list
        self.df_race["jockey_id"] = jockey_id_list

    def _get_pace(self):
        race_raptime_section = self.soup.find("section", class_="Race_Raptime")
        pace = np.nan
        if race_raptime_section:
            # "RapPace" クラスを持つspanを見つける
            rap_pace_span = race_raptime_section.find("span", class_="RapPace")

            if rap_pace_span:
                # spanの中身を取得
                pace_content = rap_pace_span.text.strip()

                if pace_content == "ペース:S":
                    pace = 1
                elif pace_content == "ペース:M":
                    pace = 2
                elif pace_content == "ペース:H":
                    pace = 3
            else:
                print("RapPace クラスを持つspanが見つかりませんでした。")
        else:
            print("Race_Raptime クラスを持つsectionが見つかりませんでした。")

        self.df_race["pace"] = [pace] * len(self.df_race)
    
    def _get_odds(self):
        self.df_odds = pd.read_html(StringIO(self.html.text))[1]

    def _set_index(self, race_id: str):
        # インデックスをrace_idにする
        self.df_race.index = [race_id] * len(self.df_race)
        self.race_results[race_id] = self.df_race
        self.df_odds.index = [race_id] * len(self.df_odds)
        self.odds_results[race_id] = self.df_odds


class PedsScraper(Scraper):
    def __init__(self):
        super().__init__()
        self.df_peds = pd.DataFrame()
        self.dict_peds = {}

    def scrape(self, horse_id_list: list):
        self.df_peds = pd.DataFrame()
        self.dict_peds = {}
        for horse_id in tqdm(horse_id_list):
            time.sleep(1)
            print(f"Scraping horse {horse_id}")
            try:
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
                html = self.session.get(url, cookies=self.session.cookies)
                html.encoding = "EUC-JP"
                df_peds_html = pd.read_html(StringIO(html.text))[0]

                # 重複を削除して1列のSeries型データに直す
                generations = {}
                for i in reversed(range(5)):
                    generations[i] = df_peds_html[i]
                    df_peds_html.drop([i], axis=1, inplace=True)
                    df_peds_html = df_peds_html.drop_duplicates()
                ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)

                self.dict_peds[horse_id] = ped.reset_index(drop=True)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        # 列名をpeds_0, ..., peds_61にする
        self.df_peds = pd.concat([self.dict_peds[key] for key in self.dict_peds], axis=1).T.add_prefix(
            "peds_"
        )

        return self.df_peds


class TodaysRaceScraper(Scraper):
    def __init__(self):
        super().__init__()
        self.data: pd.DataFrame = pd.DataFrame()
        self.shutsuba_table: pd.DataFrame = pd.DataFrame()
        self.soup: BeautifulSoup = None
        self.todays_race_time: dict = {}
        self.html: requests.models.Response = None
        self.race_id_list: list = []
        self.location_map: dict = {
            "01": "札幌",
            "02": "函館",
            "03": "福島",
            "04": "新潟",
            "05": "東京",
            "06": "中山",
            "07": "中京",
            "08": "京都",
            "09": "阪神",
            "10": "小倉",
        }

    def scrape(self, race_id: str):
        self._get_shutsuba_table(race_id)
        self._remove_space_from_columns()
        self._get_race_condition(race_id)
        self._get_pace()
        self._get_horse_jockey_id()
        self._add_index(race_id)

        return self.shutsuba_table

    def _get_shutsuba_table(self, race_id: str):
        url_shutsuba = "https://race.netkeiba.com/race/shutuba.html?race_id=" + race_id + "&rf=race_list"
        self.html = self.session.get(url_shutsuba, cookies=self.session.cookies)
        self.html.encoding = "EUC-JP"
        self.shutsuba_table = pd.read_html(StringIO(self.html.text))[0]
        self.shutsuba_table.columns = self.shutsuba_table.columns.droplevel(0)

    def _remove_space_from_columns(self):
        self.shutsuba_table.columns = self.shutsuba_table.columns.str.replace(" ", "")

    def _get_race_condition(self, race_id: str):
        self.soup = BeautifulSoup(self.html.text, "html.parser")
        texts = self.soup.find('div', attrs={'class': 'RaceData01'}).text
        texts = re.findall(r'\w+', texts)
        for text in texts:
            if 'm' in text:
                self.shutsuba_table['course_len'] = [int(re.findall(r'\d+', text)[-1])] * len(self.shutsuba_table) #20211212：[0]→[-1]に修正
            if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                self.shutsuba_table["weather"] = [text] * len(self.shutsuba_table)
            if text in ["良", "稍重", "重"]:
                self.shutsuba_table["ground_state"] = [text] * len(self.shutsuba_table)
            if '不' in text:
                self.shutsuba_table["ground_state"] = ['不良'] * len(self.shutsuba_table)
            # 2020/12/13追加
            if '稍' in text:
                self.shutsuba_table["ground_state"] = ['稍重'] * len(self.shutsuba_table)
            if '芝' in text:
                self.shutsuba_table['race_type'] = ['芝'] * len(self.shutsuba_table)
            if '障' in text:
                self.shutsuba_table['race_type'] = ['障害'] * len(self.shutsuba_table)
            if 'ダ' in text:
                self.shutsuba_table['race_type'] = ['ダート'] * len(self.shutsuba_table)
        place = race_id[4:6]
        self.shutsuba_table['place'] = [place] * len(self.shutsuba_table)

    def _get_pace(self):
        race_raptime_section = self.soup.find("section", class_="Sec_Deploy_Race")
        pace = np.nan
        if race_raptime_section:
            # "RacePace" クラスを持つdlを見つける
            rap_pace_dl = race_raptime_section.find("dl", class_="RacePace")

            if rap_pace_dl:
                # ddタグの中身を取得
                pace_content = rap_pace_dl.find("dd").text.strip()

                if pace_content == "S":
                    pace = 1
                elif pace_content == "M":
                    pace = 2
                elif pace_content == "H":
                    pace = 3
            else:
                print("RacePace クラスを持つdlが見つかりませんでした。")
        else:
            print("Sec_Deploy_Race クラスを持つsectionが見つかりませんでした。")

        self.shutsuba_table["pace"] = [pace] * len(self.shutsuba_table)

    def _get_horse_jockey_id(self):
        # horse_id
        horse_id_list = []
        horse_td_list = self.soup.find_all("td", attrs={"class": "HorseInfo"})
        for td in horse_td_list:
            horse_id = re.findall(r"\d+", td.find("a")["href"])[0]
            horse_id_list.append(horse_id)
        # jockey_id
        jockey_id_list = []
        jockey_td_list = self.soup.find_all("td", attrs={"class": "Jockey"})
        for td in jockey_td_list:
            jockey_id = re.findall(r"\d+", td.find("a")["href"])[0]
            jockey_id_list.append(jockey_id)
        self.shutsuba_table["horse_id"] = horse_id_list
        self.shutsuba_table["jockey_id"] = jockey_id_list

    def _add_index(self, race_id: str):
        self.shutsuba_table.index = [race_id] * len(self.shutsuba_table)

    def get_todays_race_id_n_time(self):
        date_str = pd.Timestamp.now().strftime("%Y%m%d")
        year, month, day = date_str[:4], date_str[4:6], date_str[6:]
        print(f"本日{year}年{month}月{day}日のレースIDを取得します")

        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode (optional)
        # Set up the WebDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        # Navigate to the URL
        url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={year}{month}{day}"
        driver.get(url)
        # Get the page source (HTML)
        html_content = driver.page_source
        # Don't forget to close the browser
        driver.quit()

        soup = BeautifulSoup(html_content, "html.parser")

        # 全てのRaceList_DataItemを見つける
        race_items = soup.find_all("li", class_="RaceList_DataItem")
        self.todays_race_time = {}
        # 各レースアイテムからrace_idとItemTimeを抽出
        for item in race_items:
            # race_idを抽出
            race_id_match = re.search(r"race_id=(\d+)", str(item))
            race_id = race_id_match.group(1) if race_id_match else None

            # ItemTimeを抽出
            item_time = item.find("span", class_="RaceList_Itemtime")
            item_time = item_time.text.strip() if item_time else None

            self.todays_race_time[race_id] = item_time
        if self.todays_race_time:
            race_id_list = list(self.todays_race_time.keys())
            race_id_list_place_kai_nichime = list(set(race_id[4:10] for race_id in race_id_list))
            for race_id in race_id_list_place_kai_nichime:
                place = self.location_map.get(race_id[:2], "Unknown")  # U
                kai = race_id[2:4]
                nichime = race_id[4:6]
                print(f"{place} {kai}回 {nichime}日目 開催")
        else:
            print("本日のレースはありません")

    # def current_race(self, race_id, month, day):
    #     race_id = str(race_id)
    #     date = '2023' + str(month).zfill(2) + str(day).zfill(2)
    #     st = ShutubaTable.scrape([race_id], date)
    #     # データ加工
    #     st.preprocessing() #前処理
    #     st.merge_horse_results(hr) #馬の過去成績結合
    #     st.merge_peds(p.peds_e) #血統データ結合
    #     st.process_categorical(r.le_horse, r.le_jockey, r.data_h) #カテゴリ変数処理
    #     today = st.data_c.copy()
    #     today.drop('date', axis=1, inplace=True)
    #     pred = lgb_clf.predict_proba(today)[:, 1]
    #     pred_data = pd.DataFrame({'pred': pred}, index=[today.馬番])

    #     return pred_data.sort_values('pred', ascending=False)

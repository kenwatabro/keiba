import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from io import StringIO

from modules.methods import Scraper


class RaceScraper(Scraper):
    def __init__(self):
        super().__init__()

        self.df_race = pd.DataFrame()
        self.race_results = {}

    def scrape(self, id_list: list) -> pd.DataFrame:

        for race_id in tqdm(id_list):
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
                self._set_index(race_id)

            # 存在しないrace_idを飛ばす
            except IndexError:
                print(f"IndexErroe: Race {url} not found")
                continue
            except (
                AttributeError
            ):  # 存在しないrace_idでAttributeErrorになるページもあるので追加
                print(f"AttributeErroe: Race {url} not found")
                continue
            # wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(e)

        # pd.DataFrame型にして一つのデータにまとめる
        race_results_df = pd.concat([self.race_results[key] for key in self.race_results])

        return race_results_df

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
            if text in ["芝", "ダート"]:
                self.df_race["race_type"] = [text] * len(self.df_race)
            if "障" in text:
                self.df_race["race_type"] = ["障害"] * len(self.df_race)
            if "m" in text:
                self.df_race["course_len"] = [int(re.findall(r"\d+", text)[-1])] * len(self.df_race)
            if text in ["良", "稍重", "重", "不良"]:
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

    def _set_index(self, race_id: str):
        # インデックスをrace_idにする
        self.df_race.index = [race_id] * len(self.df_race)
        self.race_results[race_id] = self.df_race


class PedsScraper(Scraper):
    def __init__(self):
        super().__init__()
        self.df_race = pd.DataFrame()
        self.dict_peds = {}

    def scrape(self, horse_id_list: list):
        for horse_id in tqdm(horse_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
                df_pads_html = pd.read_html(url)[0]

                # 重複を削除して1列のSeries型データに直す
                generations = {}
                for i in reversed(range(5)):
                    generations[i] = df[i]
                    df.drop([i], axis=1, inplace=True)
                    df = df.drop_duplicates()
                ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)

                self.peds_dict[horse_id] = ped.reset_index(drop=True)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        # 列名をpeds_0, ..., peds_61にする
        self.df_peds = pd.concat([self.peds_dict[key] for key in self.peds_dict], axis=1).T.add_prefix(
            "peds_"
        )
        
        return self.df_peds

# class TodayRaceScraper(Scraper):

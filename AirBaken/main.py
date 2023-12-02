import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from itertools import combinations, permutations, product
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import ast
import time
import re


class Info:
    def __init__(self):

        JST = timezone(timedelta(hours=+9), 'JST')
        self.date = datetime.now(JST)
        self.year = self.date.year
        self.month = self.date.month
        self.day = self.date.day
        self.weekday = self.date.weekday()

        self.race_url = {}

    def get_track(self):

        options = webdriver.ChromeOptions()
        # Run browser in headless mode
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')

        # Setup webdriver
        webdriver_service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=webdriver_service, options=options)

        driver.get(f'https://race.netkeiba.com/top/race_list.html?kaisai_date=' +
                   str(self.year) + str(self.month).zfill(2) + str(self.day).zfill(2))

        time.sleep(5)  # Wait for the dynamic content to load

        # Find race elements on the page
        race_elements = driver.find_elements(
            By.CSS_SELECTOR, 'dl.RaceList_DataList')

        titles = []
        for race_element in race_elements:
            title_element = race_element.find_element(
                By.CSS_SELECTOR, 'dt.RaceList_DataHeader div.RaceList_DataHeader_Top p.RaceList_DataTitle')
            # if title_element.text==None: continue
            titles.append(title_element.text)

        # Always remember to close the driver
        driver.quit()

        place_dict = {
            '札幌': '01',  '函館': '02',  '福島': '03',  '新潟': '04',  '東京': '05',
            '中山': '06',  '中京': '07',  '京都': '08',  '阪神': '09',  '小倉': '10'
        }

        def get_URL(self, kai, place, day):
            place_num = place_dict[place]
            kai = str(kai).zfill(2)
            day = str(day).zfill(2)
            self.race_url[place] = ['https://race.netkeiba.com/race/shutuba.html?race_id=' +
                                    str(self.year) + place_num + kai + day + str(i).zfill(2) for i in range(1, 13)]

        self.todays_track = []
        for kaisai in titles:
            kai, place, day = kaisai.split()
            kai = int(re.findall("\d+", kai)[0])
            day = int(re.findall("\d+", day)[0])
            get_URL(self, kai, place, day)
            self.todays_track.append(place)

    def get_horse_table(self, track, round):
        URL_list = self.race_url[track]
        round = int(re.findall("\d+", round)[0])
        URL = URL_list[round-1]
        html = requests.get(URL)
        html.encoding = "EUC-JP"
        df = pd.read_html(html.text)[0]
        df.columns = [col[0] for col in df.columns]
        waku = {}
        for i in range(len(df)):
            waku[df['馬 番'][i]] = df['枠'][i]
        df = df[['馬 番', '馬名']]

        return df, waku


# GUIウィンドウを作成
root = tk.Tk()
root.title("Bet Calculator")

# 日本時間での日付を取得
todays_race = Info()
date_str = tk.StringVar(value=todays_race.date.strftime('%Y-%m-%d'))

# 日付入力
date_entry = tk.Entry(root, textvariable=date_str)
date_entry.pack()

#競馬場
todays_race.get_track()
track_list = todays_race.todays_track
track = tk.StringVar(value=track_list[0])
track_menu = tk.OptionMenu(root, track, *track_list)
track_menu.pack()

#ラウンド
round_list = ['1R', '2R', '3R', '4R', '5R',
              '6R', '7R', '8R', '9R', '10R', '11R', '12R']
round = tk.StringVar(value=round_list[0])
round_menu = tk.OptionMenu(root, round, *round_list)
round_menu.pack()


def create_table(df, waku):
    # 新しいウィンドウを作成
    new_window = tk.Toplevel(root)
    new_window.title("Horse Table")

    # 各行に対して、テキストとチェックボックスを作成
    check_vars = []  # Checkbutton variablesを保存するリスト
    for i in range(len(df)):
        # チェックボックスを作成
        var = tk.BooleanVar()  # Checkbutton variableを作成
        check_button = tk.Checkbutton(new_window, variable=var)
        check_button.grid(row=i, column=0)  # Checkbuttonを配置
        check_vars.append(var)

        # テキストを作成
        text = " ".join(df.iloc[i].astype(str))
        text_label = tk.Label(new_window, text=text, anchor='w')
        text_label.grid(row=i, column=1, sticky='w')

        def update_menus(*args):
            if bet_method.get() in ['単勝', '複勝']:
                if buying_method.get() != '通常':
                    buying_method.set('通常')
                    buy_menu["menu"].delete(0, "end")
                    for string in buying_methods:
                        if string == '通常':
                            buy_menu["menu"].add_command(
                                label=string, command=lambda value=string: buying_method.set(value))
            else:
                if buying_method.get() == '通常':
                    buy_menu["menu"].delete(0, "end")
                    for string in buying_methods:
                        buy_menu["menu"].add_command(
                            label=string, command=lambda value=string: buying_method.set(value))

    # 買い目の方式を選択するボックス
    bet_methods = ['単勝', '複勝', 'ワイド', '馬連', '馬単', '枠連', '3連複', '3連単']
    bet_method = tk.StringVar()
    bet_method.set(bet_methods[0])
    bet_method.trace("w", update_menus)  # デフォルト値
    bet_menu = tk.OptionMenu(new_window, bet_method, *bet_methods)
    bet_menu.grid(row=len(df)+1, column=0, columnspan=2)  # print_buttonの下に配置

    # 買い方を選択するボックス
    buying_methods = ['通常', 'BOX', '流し', 'フォーメーション']
    buying_method = tk.StringVar()
    buying_method.set(buying_methods[0])  # デフォルト値
    buying_method.trace("w", update_menus)
    buy_menu = tk.OptionMenu(new_window, buying_method, *buying_methods)
    buy_menu.grid(row=len(df)+2, column=0, columnspan=2)  # bet_menuの下に配置

    # チェックボックスの状態を取得して選択された馬の名前を表示する新しいウィンドウを作成する関数
# チェックボックスの状態を取得して選択された馬の名前を表示する新しいウィンドウを作成する関数
    def show_selected_horses(bet_method=bet_method, buying_method=buying_method):
        selected_horses_window = tk.Toplevel(new_window)
        selected_horses_window.title("Selected Horses")
        selected_horses = [df.iloc[i]['馬名']
                           for i in range(len(df)) if check_vars[i].get()]
        selected_horses_numbers = [df.iloc[i]['馬 番']
                                   for i in range(len(df)) if check_vars[i].get()]

        # テキストとチェックボックスの列名を作成
        if bet_method.get() != '枠連':
            horse_number_label = tk.Label(
                selected_horses_window, text="馬番", anchor='w')
            horse_number_label.grid(row=0, column=0, sticky='w')
        elif bet_method.get() == '枠連':
            horse_number_label = tk.Label(
                selected_horses_window, text="枠番", anchor='w')
            horse_number_label.grid(row=0, column=0, sticky='w')
        horse_name_label = tk.Label(
            selected_horses_window, text="馬名", anchor='w')
        horse_name_label.grid(row=0, column=1, sticky='w')

        bet_method_dict = {'単勝': 1, '複勝': 1, 'ワイド': 2,
                           '馬連': 2, '馬単': 2, '枠連': 2, '3連複': 3, '3連単': 3}

        if buying_method.get() == '通常':
            if bet_method.get() in ['単勝', '複勝']:
                checkbox_labels = ['購入']
            elif bet_method.get() in ['ワイド', '馬連', '馬単', '枠連']:
                checkbox_labels = ['1頭目', '2頭目']
            elif bet_method.get() in ['3連複', '3連単']:
                checkbox_labels = ['1頭目', '2頭目', '3頭目']
        elif buying_method.get() == 'BOX':
            checkbox_labels = ['購入']
        elif buying_method.get() == '流し':
            checkbox_labels = ['軸', '紐']
            if bet_method.get() == '3連単':
                # 1着固定、2着固定、3着固定を選択するためのボックスを設置
                fixed_positions = ['1着固定', '2着固定', '3着固定']
                fixed_position = tk.StringVar()
                fixed_position.set(fixed_positions[0])  # デフォルト値
                fixed_position_menu = tk.OptionMenu(
                    selected_horses_window, fixed_position, *fixed_positions)
                # horse_name_labelの右に配置
                fixed_position_menu.grid(row=len(selected_horses)+2, column=1)
            if bet_method.get() == '馬単':
                fixed_positions = ['1着固定', '2着固定']
                fixed_position = tk.StringVar()
                fixed_position.set(fixed_positions[0])  # デフォルト値
                fixed_position_menu = tk.OptionMenu(
                    selected_horses_window, fixed_position, *fixed_positions)
                # horse_name_labelの右に配置
                fixed_position_menu.grid(row=len(selected_horses)+2, column=1)

        else:
            checkbox_labels = [f"{i}列目" for i in range(
                1, bet_method_dict[bet_method.get()]+1)]

        for i in range(len(checkbox_labels)):
            checkbox_label = tk.Label(
                selected_horses_window, text=checkbox_labels[i], anchor='w')
            checkbox_label.grid(row=0, column=2 + i, sticky='w')

        # 選択された馬の名前とチェックボックスを表示
        check_vars1 = []
        check_vars2 = []
        check_vars3 = []

        for i, horse_number in enumerate(selected_horses_numbers):
            horse_name = selected_horses[i]
            if bet_method.get() == '枠連':
                horse_number = waku[horse_number]
            horse_number_label = tk.Label(
                selected_horses_window, text=str(horse_number), anchor='w')
            horse_number_label.grid(row=i+1, column=0, sticky='w')

            horse_name_label = tk.Label(
                selected_horses_window, text=horse_name, anchor='w')
            horse_name_label.grid(row=i+1, column=1, sticky='w')

            if buying_method.get() == '通常':
                var1 = tk.BooleanVar()
                check_button1 = tk.Checkbutton(
                    selected_horses_window, variable=var1)
                check_button1.grid(row=i+1, column=2 + 0)
                check_vars1.append(var1)
                if bet_method_dict[bet_method.get()] >= 2:
                    var2 = tk.BooleanVar()
                    check_button2 = tk.Checkbutton(
                        selected_horses_window, variable=var2)
                    check_button2.grid(row=i+1, column=2 + 1)
                    check_vars2.append(var2)
                    if bet_method_dict[bet_method.get()] >= 3:
                        var3 = tk.BooleanVar()
                        check_button3 = tk.Checkbutton(
                            selected_horses_window, variable=var3)
                        check_button3.grid(row=i+1, column=2 + 2)
                        check_vars3.append(var3)

            elif buying_method.get() == '流し':
                var1 = tk.BooleanVar()
                check_button1 = tk.Checkbutton(
                    selected_horses_window, variable=var1)
                check_button1.grid(row=i+1, column=2 + 0)
                check_vars1.append(var1)
                var2 = tk.BooleanVar()
                check_button2 = tk.Checkbutton(
                    selected_horses_window, variable=var2)
                check_button2.grid(row=i+1, column=2 + 1)
                check_vars2.append(var2)
            elif buying_method.get() == 'BOX':
                var1 = tk.BooleanVar()
                check_button1 = tk.Checkbutton(
                    selected_horses_window, variable=var1)
                check_button1.grid(row=i+1, column=2 + 0)
                check_vars1.append(var1)
            else:
                var1 = tk.BooleanVar()
                check_button1 = tk.Checkbutton(
                    selected_horses_window, variable=var1)
                check_button1.grid(row=i+1, column=2 + 0)
                check_vars1.append(var1)
                if bet_method_dict[bet_method.get()] >= 2:
                    var2 = tk.BooleanVar()
                    check_button2 = tk.Checkbutton(
                        selected_horses_window, variable=var2)
                    check_button2.grid(row=i+1, column=2 + 1)
                    check_vars2.append(var2)
                    if bet_method_dict[bet_method.get()] >= 3:
                        var3 = tk.BooleanVar()
                        check_button3 = tk.Checkbutton(
                            selected_horses_window, variable=var3)
                        check_button3.grid(row=i+1, column=2 + 2)
                        check_vars3.append(var3)

        selected_horses_numbers = [df.iloc[i]['馬 番']
                                   for i in range(len(df)) if check_vars[i].get()]
        check_vars_all = [check_vars1, check_vars2, check_vars3]

        def show_all_bets(buying_method=buying_method, bet_method=bet_method, selected_horses_numbers=selected_horses_numbers, check_vars_all=check_vars_all):
            all_bets_window = tk.Toplevel(selected_horses_window)
            all_bets_window.title("All Bets")
            if bet_method.get() == '枠連':
                selected_horses_numbers = [waku[horse_number]
                                           for horse_number in selected_horses_numbers]

            def bet_normal(method, check_vars_all):
                num_of_horses = bet_method_dict[method]
                if num_of_horses == 1:
                    check_vars_buy1 = check_vars_all[0]
                    horse_num_list = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_buy1[i].get()]
                    horse_num = horse_num_list[0]
                    # return method+'  '+str(horse_num)
                    return [horse_num]
                elif num_of_horses == 2:
                    check_vars_buy1 = check_vars_all[0]
                    check_vars_buy2 = check_vars_all[1]
                    horse_num_list1 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_buy1[i].get()]
                    horse_num_list2 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_buy2[i].get()]
                    horse_num1 = horse_num_list1[0]
                    horse_num2 = horse_num_list2[0]
                    # return method+'  '+str(horse_num1)+' - '+str(horse_num2)
                    return [(horse_num1, horse_num2)]
                elif num_of_horses == 3:
                    check_vars_buy1 = check_vars_all[0]
                    check_vars_buy2 = check_vars_all[1]
                    check_vars_buy3 = check_vars_all[2]
                    horse_num_list1 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_buy1[i].get()]
                    horse_num_list2 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_buy2[i].get()]
                    horse_num_list3 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_buy3[i].get()]
                    horse_num1 = horse_num_list1[0]
                    horse_num2 = horse_num_list2[0]
                    horse_num3 = horse_num_list3[0]
                    # return method+'  '+str(horse_num1)+' - '+str(horse_num2)+' - '+str(horse_num3)
                    return [(horse_num1, horse_num2, horse_num3)]

            def bet_box(method,  check_vars_all):
                check_vars_buy1 = check_vars_all[0]
                horse_num_list = [selected_horses_numbers[i] for i in range(
                    len(selected_horses_numbers)) if check_vars_buy1[i].get()]
                if method in ['ワイド', '馬連', '枠連']:
                    patterns = [tuple(sorted([h1, h2]))
                                for h1, h2 in combinations(horse_num_list, 2)]

                elif method in ['馬単']:
                    patterns = [(h1, h2)
                                for h1, h2 in permutations(horse_num_list, 2)]
                elif method in ['3連複']:
                    patterns = [tuple(sorted((h1, h2, h3)))
                                for h1, h2, h3 in combinations(horse_num_list, 3)]

                elif method in ['3連単']:
                    patterns = [(h1, h2, h3)
                                for h1, h2, h3 in permutations(horse_num_list, 3)]
                return patterns

            def bet_nagashi(method, check_vars_all):
                check_vars_buy1 = check_vars_all[0]
                check_vars_buy2 = check_vars_all[1]
                jiku_num = [selected_horses_numbers[i] for i in range(
                    len(selected_horses_numbers)) if check_vars_buy1[i].get()][0]
                himo_list = [selected_horses_numbers[i] for i in range(
                    len(selected_horses_numbers)) if check_vars_buy2[i].get()]
                if method in ['ワイド', '馬連', '枠連']:
                    patterns = [tuple(sorted([jiku_num, himo]))
                                for himo in himo_list if himo != jiku_num]

                elif method in ['馬単']:
                    if fixed_position.get() == '1着固定':
                        patterns = [(jiku_num, himo) for himo in himo_list]
                    elif fixed_position.get() == '2着固定':
                        patterns = [(himo, jiku_num) for himo in himo_list]

                elif method in ['3連複']:
                    patterns = [tuple(sorted((jiku_num, himo1, himo2))) for himo1, himo2 in combinations(
                        himo_list, 2) if jiku_num != himo1 and jiku_num != himo2]

                elif method in ['3連単']:
                    patterns = []
                    # 1着が固定の場合
                    if fixed_position.get() == '1着固定':
                        for second, third in permutations(himo_list, 2):
                            patterns.append((jiku_num, second, third))
                    # 2着が固定の場合
                    elif fixed_position.get() == '2着固定':
                        for first, third in product(himo_list, repeat=2):
                            if first != third:  # 1着と3着が同じにならないようにする
                                patterns.append((first, jiku_num, third))
                    # 3着が固定の場合
                    elif fixed_position.get() == '3着固定':
                        for first, second in permutations(himo_list, 2):
                            patterns.append((first, second, jiku_num))
                return set(patterns)

            def formation(method, check_vars_all):
                if method in ['馬連', 'ワイド', '枠連']:
                    horse_num_list1 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[0][i].get()]
                    horse_num_list2 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[1][i].get()]
                    patterns = [sorted(
                        [h1, h2]) for h1 in horse_num_list1 for h2 in horse_num_list2 if h1 != h2]

                elif method in ['馬単']:
                    horse_num_list1 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[0][i].get()]
                    horse_num_list2 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[1][i].get()]
                    patterns = [
                        (h1, h2) for h1 in horse_num_list1 for h2 in horse_num_list2 if h1 != h2]

                elif method in ['3連複']:
                    horse_num_list1 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[0][i].get()]
                    horse_num_list2 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[1][i].get()]
                    horse_num_list3 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[2][i].get()]
                    patterns = [sorted([h1, h2, h3]) for h1 in horse_num_list1 for h2 in horse_num_list2 for h3 in horse_num_list3 if len(
                        {h1, h2, h3}) == 3]

                elif method in ['3連単']:
                    horse_num_list1 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[0][i].get()]
                    horse_num_list2 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[1][i].get()]
                    horse_num_list3 = [selected_horses_numbers[i] for i in range(
                        len(selected_horses_numbers)) if check_vars_all[2][i].get()]
                    patterns = [(h1, h2, h3) for h1 in horse_num_list1 for h2 in horse_num_list2 for h3 in horse_num_list3 if len(
                        {h1, h2, h3}) == 3]

                return patterns

            if buying_method.get() == '通常':
                patterns = bet_normal(bet_method.get(), check_vars_all)
            elif buying_method.get() == 'BOX':
                patterns = bet_box(bet_method.get(), check_vars_all)
            elif buying_method.get() == '流し':
                patterns = bet_nagashi(bet_method.get(), check_vars_all)
            elif buying_method.get() == 'フォーメーション':
                patterns = formation(bet_method.get(), check_vars_all)
            for i, p in enumerate(patterns):
                index_label = tk.Label(
                    all_bets_window, text=str(i+1), anchor='w')
                index_label.grid(row=i+1, column=0, sticky='w')
                pattern_label = tk.Label(
                    all_bets_window, text=str(p), anchor='w')
                pattern_label.grid(row=i+1, column=1, sticky='w')

            bet_amount = tk.Entry(all_bets_window)
            bet_amount.grid(row=0, column=2, columnspan=2)

            def record_csv(bet_method=bet_method, buying_method=buying_method, patterns=patterns, path='air_baken.csv'):
                df_air_baken = pd.read_csv(path)
                df_temp = pd.DataFrame(columns=[
                                       'date', '開催', 'R', '方式', '買い目', '購入金額', '1着', '2着', '3着', '払い戻し金額', 'race_id'])
                df_temp['買い目'] = [p for p in patterns]
                df_temp['date'] = [date_str.get()]*len(df_temp)
                df_temp['開催'] = [track.get()]*len(df_temp)
                df_temp['R'] = [round.get()]*len(df_temp)
                df_temp['方式'] = [bet_method.get()]*len(df_temp)
                df_temp['購入金額'] = [bet_amount.get()]*len(df_temp)

                r = int(re.findall("\d+", round.get())[0])
                URL_list = todays_race.race_url[track.get()]
                URL = URL_list[int(r)-1]
                race_id = re.findall("\d+", URL)[0]
                df_temp['race_id'] = [race_id]*len(df_temp)
                df_air_baken = pd.concat([df_air_baken, df_temp])
                df_air_baken.to_csv(path, index=False)
                all_bets_window.destroy()
                selected_horses_window.destroy()
                new_window.destroy()

            record_button = tk.Button(
                all_bets_window, text="Record", command=record_csv)
            record_button.grid(row=1, column=2, columnspan=2)

        show_all_bets_button = tk.Button(
            selected_horses_window, text="Show All Possible Bets", command=show_all_bets)
        show_all_bets_button.grid(
            row=len(selected_horses)+1, column=0, columnspan=2)

    show_selected_button = tk.Button(
        new_window, text="Show Selected Horses", command=show_selected_horses)
    show_selected_button.grid(row=len(df)+3, column=0, columnspan=2)


def display_data():
    selected_track = track.get()
    selected_round = round.get()
    df, waku = todays_race.get_horse_table(selected_track, selected_round)
    create_table(df, waku)


# データ表示ボタン
display_button = tk.Button(root, text="Display Data", command=display_data)
display_button.pack()


def return_calculate(path='air_baken.csv'):
    df_air_baken_all = pd.read_csv(path)
    df_air_baken_notna = df_air_baken_all[df_air_baken_all['払い戻し金額'].notna()]
    df_air_baken = df_air_baken_all[df_air_baken_all['払い戻し金額'].isna()]
    df_air_baken = df_air_baken.dropna(subset=['購入金額'])
    race_id_set = set(df_air_baken['race_id'])
    df_air_baken['1着'] = np.nan
    df_air_baken['2着'] = np.nan
    df_air_baken['3着'] = np.nan
    df_air_baken['払い戻し金額'] = np.nan

    for id in race_id_set:
        print(id)
        time.sleep(1)
        url = 'https://race.netkeiba.com/race/result.html?race_id=' + \
            str(int(id))
        html = requests.get(url)
        html.encoding = "EUC-JP"
        first, second, third = pd.read_html(html.text)[0]['馬 番'][:3]
        df_result = pd.concat(
            [pd.read_html(html.text)[1], pd.read_html(html.text)[2]])
        df_result = df_result.set_index(0)
        res_dict = {}
        houshiki_ls = ['単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '3連複', '3連単']
        if '単勝' in df_result.index:
            tansho_res_num = df_result.loc['単勝', 1]
            tansho_res_return = df_result.loc['単勝', 2].replace('円', '').split()
            for i in range(len(tansho_res_return)):
                if ',' in tansho_res_return[i]:
                    tansho_res_return[i] = tansho_res_return[i].replace(
                        ',', '')
            tansho_res = [(int(x), int(y))
                          for x, y in zip(tansho_res_num, tansho_res_return)]
            res_dict['単勝'] = tansho_res

        if '複勝' in df_result.index:
            fukusho_res_num = df_result.loc['複勝', 1].split()
            fukusho_res_return = df_result.loc['複勝', 2].replace(
                '円', '').split()
            for i in range(len(fukusho_res_return)):
                if ',' in fukusho_res_return[i]:
                    fukusho_res_return[i] = fukusho_res_return[i].replace(
                        ',', '')
            fukusho_res = [(int(x), int(y))
                           for x, y in zip(fukusho_res_num, fukusho_res_return)]
            res_dict['複勝'] = fukusho_res

        if '枠連' in df_result.index:
            wakuren_res_num = df_result.loc['枠連', 1].split()
            wakuren_res_return = df_result.loc['枠連', 2].replace(
                '円', '').split()
            for i in range(len(wakuren_res_return)):
                if ',' in wakuren_res_return[i]:
                    wakuren_res_return[i] = wakuren_res_return[i].replace(
                        ',', '')
            wakuren_res_num = [(int(wakuren_res_num[i]), int(
                wakuren_res_num[i+1])) for i in range(0, len(wakuren_res_num), 2)]
            wakuren_res = [(x, int(y))
                           for x, y in zip(wakuren_res_num, wakuren_res_return)]
            res_dict['枠連'] = wakuren_res

        if '馬連' in df_result.index:
            umaren_res_num = df_result.loc['馬連', 1].split()
            umaren_res_return = df_result.loc['馬連', 2].replace('円', '').split()
            for i in range(len(umaren_res_return)):
                if ',' in umaren_res_return[i]:
                    umaren_res_return[i] = umaren_res_return[i].replace(
                        ',', '')
            umaren_res_num = [(int(umaren_res_num[i]), int(umaren_res_num[i+1]))
                              for i in range(0, len(umaren_res_num), 2)]
            umaren_res = [(x, int(y))
                          for x, y in zip(umaren_res_num, umaren_res_return)]
            res_dict['馬連'] = umaren_res

        if '馬単' in df_result.index:
            umatan_res_num = df_result.loc['馬単', 1].split()
            umatan_res_return = df_result.loc['馬単', 2].replace('円', '').split()
            for i in range(len(umatan_res_return)):
                if ',' in umatan_res_return[i]:
                    umatan_res_return[i] = umatan_res_return[i].replace(
                        ',', '')
            umatan_res_num = [(int(umatan_res_num[i]), int(umatan_res_num[i+1]))
                              for i in range(0, len(umatan_res_num), 2)]
            umatan_res = [(x, int(y))
                          for x, y in zip(umatan_res_num, umatan_res_return)]
            res_dict['馬単'] = umatan_res

        if 'ワイド' in df_result.index:
            wide_res_num = (set(df_result.loc['ワイド', 1].split()))
            wide_res_num = combinations(wide_res_num, 2)
            wide_res_return = df_result.loc['ワイド', 2].replace('円', '').split()
            for i in range(len(wide_res_return)):
                if ',' in wide_res_return[i]:
                    wide_res_return[i] = wide_res_return[i].replace(',', '')
            wide_res = [(tuple(sorted((int(x), int(y)))), int(z))
                        for (x, y), z in zip(wide_res_num, wide_res_return)]
            res_dict['ワイド'] = sorted(wide_res, key=lambda x: x[0])

        if '3連複' in df_result.index:
            sanrenpuku_res_num = df_result.loc['3連複', 1].split()
            sanrenpuku_res_return = df_result.loc['3連複', 2].replace(
                '円', '').split()
            for i in range(len(sanrenpuku_res_return)):
                if ',' in sanrenpuku_res_return[i]:
                    sanrenpuku_res_return[i] = sanrenpuku_res_return[i].replace(
                        ',', '')
            sanrenpuku_res_num = [(int(sanrenpuku_res_num[i]), int(sanrenpuku_res_num[i+1]), int(
                sanrenpuku_res_num[i+2])) for i in range(0, len(sanrenpuku_res_num), 3)]
            sanrenpuku_res = [(x, int(y)) for x, y in zip(
                sanrenpuku_res_num, sanrenpuku_res_return)]
            res_dict['3連複'] = sanrenpuku_res

        if '3連単' in df_result.index:
            sanrentan_res_num = df_result.loc['3連単', 1].split()
            sanrentan_res_return = df_result.loc['3連単', 2].replace(
                '円', '').split()
            for i in range(len(sanrentan_res_return)):
                if ',' in sanrentan_res_return[i]:
                    sanrentan_res_return[i] = sanrentan_res_return[i].replace(
                        ',', '')
            sanrentan_res_num = [(int(sanrentan_res_num[i]), int(sanrentan_res_num[i+1]), int(
                sanrentan_res_num[i+2])) for i in range(0, len(sanrentan_res_num), 3)]
            sanrentan_res = [(x, int(y)) for x, y in zip(
                sanrentan_res_num, sanrentan_res_return)]
            res_dict['3連単'] = sanrentan_res

        df_air_baken_temp = df_air_baken[df_air_baken['race_id'] == id]
        for index, row in df_air_baken_temp.iterrows():
            row['1着'] = int(first)
            row['2着'] = int(second)
            row['3着'] = int(third)
            atari = [res[0] for res in res_dict[row['方式']]]
            if row['方式'] == '単勝':
                if int(row['買い目']) == int(first):
                    row['払い戻し金額'] = res_dict[row['方式']][0][1]*row['購入金額']//100
                else:
                    row['払い戻し金額'] = 0
            elif row['方式'] == '複勝':
                if int(row['買い目']) in [int(first), int(second), int(third)]:
                    row['払い戻し金額'] = res_dict[row['方式']][0][1]*row['購入金額']//100
                else:
                    row['払い戻し金額'] = 0
            else:
                kaime_tuple = ast.literal_eval(row['買い目'])
                if kaime_tuple in atari:
                    payout_index = atari.index(kaime_tuple)
                    row['払い戻し金額'] = res_dict[row['方式']
                                             ][payout_index][1]*row['購入金額']//100
                else:
                    row['払い戻し金額'] = 0

            df_air_baken.loc[index] = row
    df_air_baken_all = pd.concat([df_air_baken_notna, df_air_baken])
    df_air_baken_all.to_csv(path, index=False)


return_calculate_button = tk.Button(
    root, text="Return Calculate", command=return_calculate)
return_calculate_button.pack()

# GUIを表示
root.mainloop()

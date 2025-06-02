from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_driver():
    try:
        options = Options()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        logging.info('Driver initialized successfully.')
        return driver
    except Exception as e:
        logging.error(f'Error initializing driver: {e}')
        raise

def quit_driver(driver):
    driver.quit()
    logging.info('Driver quit successfully.')

def clean_folder(folder: Path):
    try:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                logging.info(f'Folder {folder} cleaned successfully.')
    except Exception as e:
        logging.error(f'Error deleting files in {folder}: {e}')

def scrap_team_stats(driver, url: str, stat_category: str):
    driver.get(url)
    time.sleep(3)

    soup = bs(driver.page_source, 'html.parser')
    # used for debugging purposes
    # with open('htmls_debug/debug_14_05.html', 'w', encoding='utf-8') as f:
    #     f.write(soup.prettify())
    # logging.debug('HTML saved for debugging.')
    try:
        tbodys = soup.find_all('tbody')
        tbody_teams = tbodys[0]
        tbody_teams_vs = tbodys[1]

        ### TIME ###
        rows = tbody_teams.find_all('tr')
        teams_stats = []

        for row in rows:
            team_data = {}

            team_cell = row.find('th', {'data-stat': 'team'})
            team_name = team_cell.find('a').text.strip() if team_cell.find('a') else team_cell.text.strip()
            team_data['time'] = team_name

            cols = row.find_all('td')
            for col in cols:
                key = col.get('data-stat')
                value = col.text.strip().replace(',', '.')
                try:
                    value = float(value)
                except ValueError:
                    pass
                team_data[key] = value

            teams_stats.append(team_data)
        logging.info(f'Found {len(teams_stats)} teams in {stat_category} stats.')
    except Exception as e:
        logging.error(f'Error parsing team stats: {e}')
        raise

    df = pd.DataFrame(teams_stats)
    df.to_excel(Path(rf'data\01_raw\teams\stats_{stat_category}_teams.xlsx'), index=False)
    logging.info(f'File {stat_category}_teams.xlsx successfully saved!')

    # ### VS TIME ###
    # try:
    #     rows_vs = tbody_teams_vs.find_all('tr')
    #     teams_stats_vs = []

    #     for row in rows_vs:
    #         team_data_vs = {}

    #         team_cell_vs = row.find('th', {'data-stat': 'team'})
    #         team_name_vs = team_cell_vs.find('a').text.strip() if team_cell.find('a') else team_cell.text.strip()
    #         team_data_vs['time'] = team_name_vs

    #         cols = row.find_all('td')
    #         for col in cols:
    #             key = col.get('data-stat')
    #             value = col.text.strip().replace(',', '.')
    #             try:
    #                 value = float(value)
    #             except ValueError:
    #                 pass
    #             team_data_vs[key] = value

    #         teams_stats_vs.append(team_data_vs)
    #     logging.info(f'Found {len(teams_stats_vs)} teams in {stat_category} vs stats.')
    # except Exception as e:
    #     logging.error(f'Error parsing team vs stats: {e}')
    #     raise

    # df_vs = pd.DataFrame(teams_stats_vs)
    # df_vs.to_excel(Path(f'data\01_raw\teams\stats_{stat_category}_teams_vs.xlsx'), index=False)

def scrap_player_stats(driver, url: str, stat_category: str):
    driver.get(url)
    time.sleep(3)

    soup = bs(driver.page_source, 'html.parser')
    tbodys = soup.find_all('tbody')
    tbody_players = tbodys[2]

    rows = tbody_players.find_all('tr', attrs={'data-row': True})
    players_stats = []

    try: 
        for row in rows:
            if row.find('td', {'data-stat': 'player'}):
                player_data = {}

                # player_cell = row.find('td', {'data-stat': 'player'})
                # player_name = player_cell.find('a').text.strip() if player_cell.find('a') else player_cell.text.strip()
                # player_data['jogador'] = player_name

                cols = row.find_all('td')
                for col in cols:
                    key = col.get('data-stat')
                    value = col.text.strip().replace(',', '.')
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    player_data[key] = value

                players_stats.append(player_data)
        logging.info(f'Found {len(players_stats)} players in {stat_category} stats.')
    except Exception as e:
        logging.error(f'Error parsing player stats: {e}')
        raise

    df = pd.DataFrame(players_stats)
    df.to_excel(Path(rf'data\01_raw\players\stats_{stat_category}_players.xlsx'), index=False)
    logging.info(f'File {stat_category}_players.xlsx successfully saved!')

def concatenate_excel_files(folder_path, output_file):
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    if not files:
        raise FileNotFoundError(f'No Excel files found in {folder_path}')
    
    if os.path.exists(output_file):
        os.remove(output_file)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for f in files:
            file_path = os.path.join(folder_path, f)
            try:
                df = pd.read_excel(file_path)
                if df.empty:
                    print(f'Warning: {file_path} is empty. Skipping.')
                    continue

                sheet_name = os.path.splitext(f)[0].replace('stats_', '').replace('_', '-').title()[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logging.info(f'File {f} concatenated!')
            except Exception as e:
                logging.error(f'Error concatenating {f}: {e}')
                raise

def main():
    players_folder = Path('data/01_raw/players')
    teams_folder = Path('data/01_raw/teams')
    players_final_path = Path('data/01_raw/players/all_stats_players.xlsx')
    teams_final_path = Path('data/01_raw/teams/all_stats_teams.xlsx')
    
    driver = initialize_driver()

    clean_folder(players_folder)
    clean_folder(teams_folder)

    urls = {
        'standard': 'https://fbref.com/pt/comps/24/stats/Serie-A-Estatisticas#all_stats_standard',
        'keeper': 'https://fbref.com/pt/comps/24/keepers/Serie-A-Estatisticas#all_stats_keeper',
        'keeper_adv': 'https://fbref.com/pt/comps/24/keepersadv/Serie-A-Estatisticas#all_stats_keeper_adv',
        'shooting': 'https://fbref.com/pt/comps/24/shooting/Serie-A-Estatisticas#all_stats_shooting',
        'passing': 'https://fbref.com/pt/comps/24/passing/Serie-A-Estatisticas#all_stats_passing',
        'passing_types': 'https://fbref.com/pt/comps/24/passing_types/Serie-A-Estatisticas#all_stats_passing_types',
        'gca': 'https://fbref.com/pt/comps/24/gca/Serie-A-Estatisticas#all_stats_gca',
        'defense': 'https://fbref.com/pt/comps/24/defense/Serie-A-Estatisticas#all_stats_defense',
        'possession': 'https://fbref.com/pt/comps/24/possession/Serie-A-Estatisticas#all_stats_possession',
        'misc': 'https://fbref.com/pt/comps/24/misc/Serie-A-Estatisticas#all_stats_misc',
        'playing_time': 'https://fbref.com/pt/comps/24/playingtime/Serie-A-Estatisticas#all_stats_playing_time',
    }

    for stat_category, url in urls.items():
        scrap_team_stats(driver, url, stat_category)
        scrap_player_stats(driver, url, stat_category)

    quit_driver(driver)

    concatenate_excel_files(players_folder, players_final_path)
    concatenate_excel_files(teams_folder, teams_final_path)

if __name__ == "__main__":
    main()
    logging.info('Data extraction completed successfully!')
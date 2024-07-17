import glob
import pandas as pd
from tqdm import tqdm
import pathlib
from cleantext import clean
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time
from datasets import load_dataset


def refreshing_my_token():
    refresh_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTcyMTMwOTE5MiwiaWF0IjoxNzIxMjIyNzkyLCJqdGkiOiIzOWU1MzNmNDI4ZGY0NGQ4YTQ5NjgyYWM5NGZiMWJlNiIsInVzZXJfaWQiOjF9.PRdOsEKDT8gLE7pyNt6NXSgKxc68wH89Wq6TG4dUCmo"
    url = "https://trans.uicgroup.tech/api/v1/users/TokenRefresh/"
    headers = {'Content-Type': 'application/json'}
    data = {'refresh': refresh_token}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        new_token = response.json().get('access')
        return new_token
    else:
        return None

def clean_text(text):
    return clean(
        text=text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=False,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        lang="en"
    ).replace('""', '"').replace("'", '`').replace("\n", '')

def translate_batch(texts, token, session):
    if texts:
        url = 'https://trans.uicgroup.tech/api/v1/trans/translate/'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        }
        data = {
            'text': " ".join(texts),
            'source_language': 'en',
            'target_language': 'uz'
        }
        response = session.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get('translated_text', '').split('. ')
        elif response.status_code in [401, 403]:
            token = refreshing_my_token()
            if token:
                return translate_batch(texts, token, session)
    return ["none"] * len(texts)

def process_row(row, token, session):
    sentences = [clean_text(sentence) + "." for sentence in str(row).split(".") if sentence.strip()]
    return translate_batch(sentences, token, session)

def save_to_csv(content, file_name, save_row, total_count):
    pathlib.Path(f'./Databricks_dolly_15K_translated/{file_name}').mkdir(exist_ok=True, parents=True)
    df_out = pd.DataFrame(data=content)
    df_out.to_csv(f'./Databricks_dolly_15K_translated/{file_name}/{file_name}_{save_row-99}-{save_row}_rows_data_total_{save_row//100}-{total_count}.csv', index=False)


def main():

    ds = load_dataset("databricks/databricks-dolly-15k")
    df = pd.DataFrame(ds['train'])
    file_name = 'translated'
    total_count = len(df) // 100
    save_row = 1
    content = {'instruction': []}

    with requests.Session() as session:
        token = refreshing_my_token()
        if not token:
            print("Failed to start the translation process due to token refresh failure.")
            
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_row = {executor.submit(process_row, row, token, session): row for row in df['instruction']}
            for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
                result = future.result()
                if result:
                    content['instruction'].extend(result)
                    if save_row % 100 == 0:
                        save_to_csv(content, file_name, save_row, total_count)
                        content = {'instruction': []}
                save_row += 1

        if content['instruction']:
            save_to_csv(content, file_name, save_row-1, total_count)

if __name__ == "__main__":
    main()

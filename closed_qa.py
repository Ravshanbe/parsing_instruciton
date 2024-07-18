import glob
import pandas as pd
from tqdm import tqdm
import pathlib
from cleantext import clean
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time
from datasets import load_dataset

def refresh_tokenrefresher():
    url = "https://trans.uicgroup.tech/api/v1/users/login/"
    headers = {'Content-Type': 'application/json'}
    data = {
        "username": "admin",
        "password": "pQ4kW1xW8rR3yE6l"}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        refresh_token = response.json().get('refresh')
        return refresh_token
    else:
        print(response.status_code)
        return None

def refreshing_my_token():
    refresh_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTcyMTM2OTU3MSwiaWF0IjoxNzIxMjgzMTcxLCJqdGkiOiJmMTVhOTk0OTc3Mjg0MDhkOTJlN2ZiNjEzMGI4OGQ0NyIsInVzZXJfaWQiOjF9.kVqBA20Qb5BqHUE07WYGzH5WFthiqLF_VJ6FIGyBGFA"
    url = "https://trans.uicgroup.tech/api/v1/users/TokenRefresh/"
    headers = {'Content-Type': 'application/json'}
    data = {'refresh': refresh_token}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        new_token = response.json().get('access')
        return new_token
    elif response.status_code == 401:
        refresh_token = refresh_tokenrefresher()
        data = {'refresh': refresh_token}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
          new_token = response.json().get('access')
          return new_token
        else:
          print(response.status_code)
          return None

    else:
        print(response.status_code)
        return None


def clean_text(text):
    return clean(
    text = text,
    fix_unicode=True,          
    to_ascii=False,           
    lower=False,              
    no_urls=False,            
    no_emails=False,          
    no_phone_numbers=False,   
    no_numbers=False,          
    no_digits=False,          
    no_currency_symbols=False,
    no_punct=False,           
    lang="en"                       
).replace('""', '"').replace("'", '`')

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
    sentences_instruction = [clean_text(sentence) + "." for sentence in str(row['instruction']).split(".") if sentence.strip()]
    sentences_context = [clean_text(sentence) + "." for sentence in str(row['context']).split(".") if sentence.strip()]
    sentences_response = [clean_text(sentence) + "." for sentence in str(row['response']).split(".") if sentence.strip()]
    translated_instruction = " ".join(translate_batch(sentences_instruction, token, session))
    translated_context = " ".join(translate_batch(sentences_context, token, session))
    translated_response = " ".join(translate_batch(sentences_response, token, session))
    return {'instruction': translated_instruction, 'context': translated_context,'response': translated_response}

def save_to_csv(content, file_name, save_row, total_count):
    pathlib.Path(f'./Databricks_dolly_15K_/{file_name}').mkdir(exist_ok=True, parents=True)
    df_out = pd.DataFrame(data=content)
    df_out.to_csv(f'./Databricks_dolly_15K_/{file_name}/{file_name}-{save_row//100}-{total_count}.csv', index=False)





def main():
    ds = load_dataset("databricks/databricks-dolly-15k")
    df = pd.DataFrame(ds['train'])
    df = df[df.category == 'closed_qa']
    file_name = 'closed_qa'
    total_count = len(df) // 100
    save_row = 1
    content = {'instruction': [], 'context': [],'response': []}

    with requests.Session() as session:
        token = refreshing_my_token()
        if not token:
            print("Failed to start the translation process due to token refresh failure.")
            return

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_row = {executor.submit(process_row, row, token, session): row for _, row in df.iterrows()}
            for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
                result = future.result()
                if result:
                    content['instruction'].append(result['instruction'])
                    content['context'].append(result['context'])
                    content['response'].append(result['response'])

                    if save_row % 5 == 0:
                        print(content['instruction'][-1:], content['context'][-1:],content['response'][-1:])
                        save_to_csv(content, file_name, save_row, total_count)
                        content = {'instruction': [], 'context': [], 'response': []}
                save_row += 1

        if content['response']:
            save_to_csv(content, file_name, save_row-1, total_count)

if __name__ == "__main__":
    main()

import os

def crawl_twitter(token, keyword, start_date, end_date, limit):
    query = f"{keyword} since:{start_date} until:{end_date}"
    output_file = "DataPenelitian.csv"
    
    os.makedirs("Data", exist_ok=True)

    cmd = (
        f'npx -y tweet-harvest@2.6.1 '
        f'-o "{output_file}" '
        f'-s "{query}" '
        f'--tab "LATEST" '
        f'-l {limit} '
        f'--token {token}'
    )

    print("Menjalankan perintah:\n", cmd)
    os.system(cmd)
    print(f"Data berhasil disimpan di {output_file}")
    return output_file
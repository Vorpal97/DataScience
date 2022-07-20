import os
import etl
folder = './grafici_tsa'
if not os.path.exists(folder):
    os.mkdir(folder)
df = etl.importa_tsa('./dataset/merged_top_100_cry.csv')

'''
coins_to_keep = ['bitcoin', 'ethereum', 'dogecoin', 'solana']
#ritorna dizionario con le crypto selezionate
df_list = etl.create_crypto_dict(df, coins_to_keep)

with open(folder + '/means.txt', 'w') as f:
        f.write(str(df))'''





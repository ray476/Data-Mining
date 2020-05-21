import pandas as pd

df = pd.read_csv('rose_genotypes.txt', sep='\t', header=None, names=['genes', 'color'])
# print(df.to_string())
arr = []
for row in df.itertuples(index=False):
    if (row[0][:2] == 'rr') and (row[0][-2:] == 'ss'):
        # if(row[0][2:4] != 'YY' and row[0][2:4] != 'yy'):
        #     if (row[0][4:6] != 'WW' and row[0][4:6] != 'ww'):
                arr.append((row[0], row[1]))

print(arr)
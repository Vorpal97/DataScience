from matplotlib import pyplot as plt
import pandas as pd
import etl

df = etl.importa()
s = df['type_1'].value_counts()

dict = s.to_dict()

x = list(dict.keys())
y = list(dict.values())


plt.bar(x, y)
plt.xticks(rotation=90)
plt.suptitle('Conteggio tipi', fontsize=20)
plt.xlabel('Tipo', fontsize=18)
plt.ylabel('Numero', fontsize=16)
plt.show()
plt.savefig('/Users/simonecappella/Desktop/test.jpg')



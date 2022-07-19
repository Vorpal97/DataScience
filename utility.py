import etl

df = etl.importa()

print(df.columns.get_loc('base_experience'))
print(df.columns.get_loc('total_points'))



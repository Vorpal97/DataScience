from scipy.io.arff import loadarff
import pandas as pd
raw_data = loadarff('./dataset/AbnormalHeartbeat_TRAIN.arff')
df = pd.DataFrame(raw_data[0])
print(df)
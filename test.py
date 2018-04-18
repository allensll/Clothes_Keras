import pandas as pd

df = pd.DataFrame(columns=['a','b'])

df.add(pd.DataFrame([1,2]))
print(df)
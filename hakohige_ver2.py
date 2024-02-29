import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Assuming you have loaded your data into df1
df1 = pd.read_csv("./env_comp_test/Ant_nomal_100000_weights_testNomal.csv")
df2 = pd.read_csv("./env_comp_test/Ant_nomal_100000_weights_testRandom.csv")
df3 = pd.read_csv("./env_comp_test/Ant_nomal_100000_weights_testAdv.csv")
df4 = pd.read_csv("./env_comp_test/Ant_random_0515_09_100000_weights_testNomal.csv")
df5 = pd.read_csv("./env_comp_test/Ant_random_0515_09_100000_weights_testRandom.csv")
df6 = pd.read_csv("./env_comp_test/Ant_random_0515_09_100000_weights_testAdv.csv")
df7 = pd.read_csv("./env_comp_test/Ant_adv_0515_09_100000_weights_testNomal.csv")
df8 = pd.read_csv("./env_comp_test/Ant_adv_0515_09_100000_weights_testRandom.csv")
df9 = pd.read_csv("./env_comp_test/Ant_adv_0515_09_100000_weights_testAdv.csv")
df10 = pd.read_csv("./env_comp_test/Ant_advF_lr6_10_10000_weights_testNomal.csv")
df11 = pd.read_csv("./env_comp_test/Ant_advF_lr6_10_10000_weights_testRandom.csv")
df12 = pd.read_csv("./env_comp_test/Ant_advF_lr6_10_10000_weights_testAdv.csv")
# 各データフレームにカテゴリ列を追加
#Train
df1['Train'] = 'Normal Train'
df2['Train'] = 'Normal Train'
df3['Train'] = 'Normal Train'

df4['Train'] = 'Random Train'
df5['Train'] = 'Random Train'
df6['Train'] = 'Random Train'

df7['Train'] = 'ADA Train'
df8['Train'] = 'ADA Train'
df9['Train'] = 'ADA Train'

df10['Train'] = 'AFT Train'
df11['Train'] = 'AFT Train'
df12['Train'] = 'AFT Train'
#Test
df1['Test environment'] = 'Normal'
df2['Test environment'] = 'Random'
df3['Test environment'] = 'Adversarial'

df4['Test environment'] = 'Normal'
df5['Test environment'] = 'Random'
df6['Test environment'] = 'Adversarial'

df7['Test environment'] = 'Normal'
df8['Test environment'] = 'Random'
df9['Test environment'] = 'Adversarial'

df10['Test environment'] = 'Normal'
df11['Test environment'] = 'Random'
df12['Test environment'] = 'Adversarial'

# 仮定: 各データフレームの'Value'列に注目する
columns_of_interest = ['reward', 'Train','Test environment']

# 全てのデータフレームを結合
combined_df = pd.concat([
    df1[columns_of_interest], 
    df2[columns_of_interest], 
    df3[columns_of_interest],
    df4[columns_of_interest], 
    df5[columns_of_interest], 
    df6[columns_of_interest],
    df7[columns_of_interest], 
    df8[columns_of_interest], 
    df9[columns_of_interest],
    df10[columns_of_interest], 
    df11[columns_of_interest], 
    df12[columns_of_interest]
])
# 箱ひげ図のプロット
palette = ['b', 'g', 'r']
plt.figure(figsize=(15, 8))
sns.boxplot(x='Train', y='reward',hue='Test environment', data=combined_df, sym="",showmeans=True,  palette=palette,
            meanprops={"marker":"o", "markerfacecolor":"white", 
                       "markeredgecolor":"black", "markersize":"10"})
plt.legend(loc="upper right")
plt.legend(title="Test environment")
# 軸ラベルの設定
plt.xlabel('Training Methods', fontsize=14, fontweight='bold')
plt.ylabel('Reward', fontsize=14, fontweight='bold')
#plt.xticks(rotation=45)  # x軸のラベルを45度回転
plt.ylim(-1000, 7500)
plt.savefig("testAnt_2.pdf")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#将所有数据组装到一个DataFrame里面
years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = '../babynames/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)

    frame['year'] = year
    pieces.append(frame)
    
names = pd.concat(pieces, ignore_index=True)

#做数据透视，统计每年分性别的出生数量
total_births = names.pivot_table('births', index='year',\
				columns='sex',aggfunc=sum)
total_births.tail()
#print(total_births.tail())

total_births.plot(title='Total births by sex and year')

#plt.show()
#plt.savefig('../png/Total birth by sex and year.png')

#插入prop列，存放指定名字的婴儿数相对于总数的比例
def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group

names = names.groupby(['year','sex']).apply(add_prop)

print(np.allclose(names.groupby(['year','sex']).prop.sum(),1))

#取出子集，每对sex/year组合的前1000个名字，分组操作
def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)



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

#print(np.allclose(names.groupby(['year','sex']).prop.sum(),1))

#取出子集，每对sex/year组合的前1000个名字，分组操作
def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)

#将前1000个名字分为男女两部分
boys = top1000[top1000.sex=='M']
girls = top1000[top1000.sex=='F']

#透视表，按year和name统计的总出生数透视表
total_births = top1000.pivot_table('births', index='year', \
				columns='name', aggfunc=sum)
#print(total_births)

#绘制几个名字的曲线
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12,10), grid=False,\
            title='Number of births per year')
#plt.show()

#评估命名多样性的增长
#计算最流行的1000个名字所占的比例变化
table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
table.plot(title='Sum of table 1000 prop by year and sex', \
           yticks=np.linspace(0,1.2,13), xticks=range(1880, 2020, 10))
#plt.show()

#另一个办法是计算占总出生人数前50%的不同名字的数量
dfa = boys[boys.year == 2010]
prop_cumsum = dfa.sort_index(by='prop', ascending=False).prop.cumsum()
a=prop_cumsum.searchsorted(0.5)
print(a)

dfb = boys[boys.year == 1900]

prop_cumsum_b = dfb.sort_index(by='prop', ascending=False).prop.cumsum()

b=prop_cumsum_b.searchsorted(0.5)
print('1900 %d'%b)

#对所有的year/sex组合执行这个运算
#获取占50%的名字
def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().searchsorted(q)[0] + 1
#diversity时每一年的占50%的名字列表
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')

diversity.plot(title='Number of popular names in top 50%')

#plot.show()

#从name列获取最后一个字母
get_last_letter = lambda x:x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table('births', index=last_letters,\
                      columns=['sex','year'], aggfunc=sum)
#选出具有代表性的三年，并输出前面几行
subtable = table.reindex(columns=[1910,1960,2010], level='year')
print(subtable.head())

print(subtable.sum())

letter_prop = subtable / subtable.sum().astype(float)

fig, axes = plt.subplots(2,1,figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male', legend=False)
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Femal')
#plt.show()

letter_prop = table / table.sum().astype(float)
dny_ts = letter_prop.ix[['d','n','y'],'M'].T
print(dny_ts.head())
dny_ts.plot()
#plt.show()

#分析男孩女孩名字的转变找出lesl开头的一组名字
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
print(lesley_like)

#利用这个结果过滤其他名字，并按名字分组计算出生数以查看相对频率
filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered.groupby('name').births.sum())

table = filtered.pivot_table('births', index='year', columns='sex',\
				aggfunc='sum')
table = table.div(table.sum(1), axis=0)
print(table.tail())
table.plot(style={'M':'k-', 'F':'k--'})
plt.show()


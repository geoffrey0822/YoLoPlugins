import os,sys,leveldb
import random

db=leveldb.LevelDB(sys.argv[1])
count=0
print 'Counting data in DB...',
for key,value in db.RangeIter():
    count+=1
print '%d[Done]'%count
ord_list=range(0,count)
if len(sys.argv)>2 and sys.argv[2]=='shuffle':
    random.shuffle(ord_list)
for ordi in ord_list:
    print ordi,
    print '->',
    data=db.Get(str(ordi),default=None)
    print data
    if ordi>33:
        break
#coding:utf-8

import sys
import json
import jieba
from datetime import datetime, timedelta
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.insert(0, '/opt/tiger/nlp')
sys.path.insert(0, '/opt/tiger/pyutil')
from nlp_util.program.hive_query import execute_hive_query

def cut_text(text):
    seg_list = jieba.cut(text)
    return " ".join(seg_list)

def get_all_title_data():
    etime = (datetime.now()-timedelta(days=1)).strftime('%Y%m%d')
    stime = (datetime.now()-timedelta(days=721)).strftime('%Y%m%d')
    sql = '''
            select
                title
            from
                dm_content.content_group_all
            where
                date = '%s'
                and from_unixtime(
                    int(fetch_time),
                    'yyyyMMdd'
                ) >= '%s'
    ''' % (etime, stime)
    res = execute_hive_query('get_pid_tasks', 'kehuangsong', sql, accept_empty=True)
    print 'title data loaded: %s' % len(res)
    with open('./data/title_data.txt', 'w') as f:
        for re in res:
            try:
                title = re[0]
                split_title = cut_text(title)
                f.write('%s\n' % (split_title.encode("utf-8")))
            except:
                continue

if __name__ == '__main__':
    get_all_title_data()

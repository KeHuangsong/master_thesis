#coding:utf-8

import re
import sys
import json
import jieba
from datetime import datetime, timedelta
reload(sys)
sys.setdefaultencoding('utf-8')
from nlp_util.program.hive_query import execute_hive_query


class GetData(object):
    def clean_str(self, string):
        pattern = re.compile(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])")
        cstr = re.sub(pattern, '', string)
        return cstr

    def cut_text(self, text):
        # text = self.clean_str(text)
        seg_list = jieba.cut(text)
        return " ".join(seg_list)

    def get_title_data(self, days=15, data_size=10000, category_num=20):
        sql = '''
                select
                    distinct a.category,
                    title
                from
                    (
                        select
                            category,
                            title,
                            row_number() over (
                                partition by category
                                order by
                                    rand()
                            ) as rank
                        from
                            dm_content.content_group_all
                        where
                            date = '${date}'
                            and media_id > 0
                            and mp_id = 54
                            and group_source = 2
                            and category > '0'
                            and composition & 384 > 0
                            and from_unixtime(
                                int(fetch_time),
                                'yyyyMMdd'
                            ) > 'start_time'
                    ) a
                    join (
                        select
                            category,
                            num
                        from(
                                select
                                    category,
                                    count(distinct group_id) as num
                                from
                                    dm_content.content_group_all
                                where
                                    date = '${date}'
                                    and media_id > 0
                                    and mp_id = 54
                                    and group_source = 2
                                    and category > '0'
                                    and composition & 384 > 0
                                    and from_unixtime(
                                        int(fetch_time),
                                        'yyyyMMdd'
                                    ) > 'start_time'
                                group by
                                    category
                            ) a
                        order by
                            num desc
                        limit
                            category_num
                    ) b on a.category = b.category
                where
                    rank < data_size
        '''.replace('${date}', (datetime.now()-timedelta(days=2)).strftime('%Y%m%d'))\
           .replace('start_time', (datetime.now()-timedelta(days=days+1)).strftime('%Y%m%d'))\
           .replace('data_size', str(data_size))\
           .replace('category_num', str(category_num))
        res = execute_hive_query('get_category_title', 'kehuangsong', sql, accept_empty=True)
        print 'title data loaded: %s' % len(res)
        with open('./data/raw_title_data.txt', 'w') as f:
            for row in res:
                try:
                    category = row[0]
                    title = row[1]
                    split_title = self.cut_text(title)
                    f.write('%s\t%s\n' % (category, split_title.encode("utf-8")))
                except:
                    continue


if __name__ == '__main__':
    get_data = GetData()
    get_data.get_title_data(days=15, data_size=10000, category_num=20)

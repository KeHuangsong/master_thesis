#coding:utf-8

import re
import sys
import json
import jieba
from datetime import datetime, timedelta
reload(sys)
sys.setdefaultencoding('utf-8')
from nlp_util.program.hive_query import execute_hive_query


def clean_str(string):
    pattern = re.compile(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])")
    cstr = re.sub(pattern, '', string)
    return cstr


def cut_text(text):
    # text = clean_str(text)
    seg_list = jieba.cut(text)
    return " ".join(seg_list)


def get_title_data():
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
                        ) > '${date-7}'
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
                                ) > '${date-7}'
                            group by
                                category
                        ) a
                    order by
                        num desc
                    limit
                        20
                ) b on a.category = b.category
            where
                rank < 10001
    '''.replace('${date}', (datetime.now()-timedelta(days=1)).strftime('%Y%m%d'))\
       .replace('${date-7}', (datetime.now()-timedelta(days=8)).strftime('%Y%m%d'))
    res = execute_hive_query('get_category_title', 'kehuangsong', sql, accept_empty=True)
    print 'title data loaded: %s' % len(res)
    with open('./data/raw_title_data.txt', 'w') as f:
        for row in res:
            try:
                category = row[0]
                title = row[1]
                split_title = cut_text(title)
                f.write('%s\t%s\n' % (category, split_title.encode("utf-8")))
            except:
                continue


if __name__ == '__main__':
    get_title_data()

import csv
import feedparser
import re
from collections import Counter

feeds = {
    "mashable": "http://feeds.feedburner.com/Mashable",
    "techcrunch": "http://feeds.feedburner.com/TechCrunch/social",
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "rt": "https://www.rt.com/rss/news/",
    "nasa": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "wired": "https://www.wired.com/feed/",
    "boingboing": "http://feeds.feedburner.com/boingboing/ibag",
    "usatoday": "http://rssfeeds.usatoday.com/usatoday-NewsTopStories",
    "buzzfeed": "https://www.buzzfeed.com/index.xml",
    "gizmodo": "http://feeds.gawker.com/gizmodo/full",
    "techradar": "http://www.techradar.com/rss",
    "ft_usa": "http://www.ft.com/rss/home/us",
    "ft_eu": "http://www.ft.com/rss/home/europe",
    "ft_asia": "http://www.ft.com/rss/home/asia",
    "jp_times": "http://www.japantimes.co.jp/feed/topstories/"
}


def getwords(html):
    txt=re.compile(r'<[^>]+>').sub('',html)
    words=re.compile(r'[^A-Z^a-z]+').split(txt)
    return [word.lower() for word in words if word != '']


def get_wordcount(feed):
    wordcount = Counter()
    fd = feedparser.parse(feed)
    for e in fd.entries:
        if 'summary' in e:
            text = e.summary
        else:
            text = e.description
        words = getwords(text + ' ' + e.title)
        wordcount.update(words)
    return wordcount


word_uniqueness = {}
out = {}
for feed_name, feed in feeds.items():
    counts = get_wordcount(feed)
    out[feed_name] = dict(get_wordcount(feed))
    for word, count in counts.items():
        word_uniqueness.setdefault(word, 0)
        if count > 1:
            word_uniqueness[word] += 1

word_list = []
for word, count in word_uniqueness.items():
    frac = count / len(feeds)
    if frac > 0.1 and frac < 0.6:
        word_list.append(word)

word_list.insert(0, 'blog')
with open('rss_word_count.csv','w') as file:
    csv_file = csv.writer(file, delimiter=',')
    csv_file.writerow(word_list)
    for feed, wc in out.items():
        row = [feed]
        for word in word_list[1:]:
            if word in wc:
                row.append(wc[word])
            else:
                row.append(0)
        csv_file.writerow(row)

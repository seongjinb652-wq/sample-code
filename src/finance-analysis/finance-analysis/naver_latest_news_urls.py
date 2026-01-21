import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import tavily_search_urls as ts

def news_latest_url(query):
    tavily_search = False
    #fianl_query = query + " 실적"
    url = 'https://search.naver.com/search.naver?where=news&ie=utf8&sm=nws_hty&query={}'.format(query)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    r = requests.get(url, headers=headers)  # 웹페이지 요청

    html = r.text  # r.content Or r.text : 웹페이지 형태
    #print(html)  #  웹페이지 정보 획득
    #
    soup = BeautifulSoup(html, 'lxml')  # 파서 지정

    #  해당 div 찾기 (클래스 3개 모두 일치)
    container = soup.find(
        "div",
        #class_="P6OUecgVB94hFNtoYv9U desktop_mode api_subject_bx"
        class_="sds-comps-vertical-layout sds-comps-full-layout fds-news-item-list-tab"
        #class_="sds-comps-vertical-layout sds-comps-full-layout"
    )
    #print(container)
    print("naver news data crawling....")
    if container is None:  # 네이버 크롤링 문제일 경우 tavily 로 대체
        print("[WARN] naver news container not found | query={}".format(query))
        print("Replace with Tavily Search!!")
        tavily_search = True
        hrefs = ts.tavily_search(query)
        return hrefs ,  tavily_search # ← 핵심: 빈 리스트 반환

    #  div 내부의 모든 a 태그 href 추출
    hrefs = [
        a["href"]
        for a in container.find_all("a", href=True)
    ]
    #print(hrefs)

    return hrefs, tavily_search


def clean_urls(urls):
    unique_urls = set()
    cleaned = []

    for u in urls:
        if not u:
            continue

        # http / https 로 시작하지 않으면 제거
        if not u.startswith(("http://", "https://")):
            continue

        # '#' 만 있는 링크 제거
        if u.strip() == "#":
            continue

        # 중복 제거
        if u not in unique_urls:
            unique_urls.add(u)
            cleaned.append(u)

    return cleaned[3:]



def normalize_news_urls(urls: list[str]) -> list[str]:
    domain_articles = {}   # domain -> article_url
    domain_roots = {}      # domain -> root_url

    ARTICLE_KEYWORDS = (
        "article",
        "articleView",
        "NewsView",
        "view",
        "news/",
        "mnews"
    )

    for u in urls:
        if not u.startswith(("http://", "https://")):
            continue

        parsed = urlparse(u)
        domain = parsed.netloc
        path = parsed.path.lower()

        # 기사 URL 판별
        is_article = any(k.lower() in path for k in ARTICLE_KEYWORDS)

        if is_article:
            # 같은 도메인이면 기사 URL 우선 저장
            domain_articles[domain] = u
        else:
            # 루트 또는 비기사 URL 저장
            domain_roots.setdefault(domain, u)

    # 결과 조합
    result = []

    for domain in set(domain_articles) | set(domain_roots):
        if domain in domain_articles:
            result.append(domain_articles[domain])
        else:
            result.append(domain_roots[domain])

    return result

if __name__ == "__main__":
    query = 'RFHIC'
    urllist, tavily_used  = news_latest_url(query)
    print(urllist, tavily_used)
    if tavily_used == False:
        cleaned_urls = clean_urls(urllist) # url이 아닌 정보나 중복 제거
    else:
        cleaned_urls = urllist
    print('='*80)
    print(cleaned_urls)
    normalized_urls = normalize_news_urls(cleaned_urls) # 도메인 url이 동일 정보 제거
    print('='*80)
    print(normalized_urls)

#
# # 위 방법과 다른 find_all(), find() 메서드 활용
# find_all() ==> 찾은 tag 정보를 list 형태로 반환

# urls = [a.get("href") for a in soup.find_all("a")]
# print(urls)

# newtitlesoup = soup.find_all(class_ = 'sds-comps-vertical-layout sds-comps-full-layout fds-news-item-list-tab')
# print(newtitlesoup)
#
# urls = []
#
# for container in newtitlesoup:
#     for a in container.find_all("a", href=True):
#         urls.append(a["href"])
#
# print(list(set(urls)))
#print( newtitlesoup[0].get('href') )

# urls = [a.get("href") for a in newtitlesoup if a.get("href")]
# print(urls)

# # print(len(newtitlesoup))
# # print(newtitlesoup[0].text)
# newtitlelist = []
# for news in newtitlesoup:
#     print( news )
#     newtitlelist.append(news.text)
# print(newtitlelist)
#
# import pandas as pd
# newsdf = pd.DataFrame({'NewsTitle':newtitlelist})
# print(newsdf)
#
# newsdf.to_excel("newstitle_new.xlsx")
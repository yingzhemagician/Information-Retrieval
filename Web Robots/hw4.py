import re
from bs4 import BeautifulSoup
from urllib import parse, request
from queue import PriorityQueue
import ssl


def get_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)

site = 'https://cs.jhu.edu/~yarowsky'
# domain = parse.urlparse(site).netloc
# if domain[:4] == "www.":
#     domain == domain[3:]

context = ssl._create_unverified_context()
r = request.urlopen(site, context=context)

for l in get_links(site, r.read()):
    print(l)

print("\n\n\n")

def print_non_local_links(root):
    # TODO
    domain = parse.urlparse(root).netloc
    context = ssl._create_unverified_context()
    r = request.urlopen(root, context=context)
    for l in get_links(root, r.read()):
        if domain not in parse.urlparse(l[0]).netloc:
            print("non-local", l)
        elif '#' in l[0]:
            print("self-refe", l)
        else:
            print("         ", l)

#print_non_local_links('https://cs.jhu.edu/~winston/ir/nonlocal.html')
print_non_local_links('https://cs.jhu.edu/~yarowsky')


print("\n\n\n")


def crawl(root, extract=lambda a, h: None):
    # TODO
    visited = set()
    results = []


    def shouldvisit(address):
        return "cs.jhu.edu" in address and address not in visited


    def wanted(req):
        return 'text/html' in req.headers['Content-Type'] or 'application/pdf' in req.headers['Content-Type']

    def relevance(title):
        if "Homework" in title:
            return 1
        elif "Undergraduate" in title:
            return 2
        elif "PhD" in title:
            return 3
        else:
            return 4

    queue = PriorityQueue()
    queue.put((1, root))

    level = 1

    while not queue.empty():
        priority, address = queue.get()
        if len(results) >= 50:
            print("stop crawl")
            break

        try:
            context = ssl._create_unverified_context()
            r = request.urlopen(address, context=context)
            if r.status == 200:#success
                visited.add(address)

                if wanted(r):
                    results.append(address)

                html = r.read()
                extract(address, html)

                print(address)

                for link, title in get_links(address, html):
                    if '#' in link:
                        continue

                    if shouldvisit(link):
                        rel = relevance(title)

                        print("           level = ", level, " ", link, "  ", title, "   rel = ", rel)
                        queue.put((rel, link))

                level = level + 1

        except Exception as e:
            print(e, address)

    print("Crawl Finished!")
    return visited, results

the_visited, the_result = crawl("https://cs.jhu.edu/~yarowsky/cs466.html")

# for each_item in the_result:
#     print(each_item)


def extract(address, html):
    # TODO
    for match in re.findall('\d\d\d-\d\d\d-\d\d\d\d', str(html)):
        print('; '.join([address, 'PHONE', match]))

    for match in re.findall("[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}", str(html)):
        if match[-4:] != ".png" and match[-5:] != ".jpeg" and match[-4:] != ".gif":
            print('; '.join([address, 'EMAIL', match]))

    for match in re.findall("\w+,\s*\w+\s[0-9]{5}", str(html)):
        print('; '.join([address, 'CITY', match]))

visited = crawl("https://cs.jhu.edu/~yarowsky", extract)
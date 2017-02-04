import csv
from bs4 import BeautifulSoup

reader = csv.reader(open('./Svg/unemployment-aug2010.txt', 'r'), delimiter = ",")
svg = open("./Svg/usa_counties.svg", 'r').read()

unemployment = {}
for row in reader:
    try:
        full_fips = row[1] + row[2]
        rate = float(row[5])
        unemployment[full_fips] = rate
    except:
        continue

soup = BeautifulSoup(svg, 'xml')
paths = soup.findAll('path')

colors =['#f1eef6','#d4b9da','#c994c7','#df65b0','#dd1c77','#980043']

path_style="font-size:12px;fill-rule:nonzero;stroke:#000000;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:"

for p in paths:
    if p['id'] not in ["State_Lines", "separator"]:
        try:
            rate = unemployment[p['id']]
        except:
            continue
        if rate > 10: color_class = 5
        elif rate > 8: color_class = 4
        elif rate > 6: color_class = 3
        elif rate > 4: color_class = 2
        elif rate > 2: color_class = 1
        else: color_class = 0
        color = colors[color_class]
        p['style'] = path_style + color

print(soup.prettify())
import requests
import lxml.html as l
import csv

properties = {"Напряжение аккумулятора": ("В", float),
              "Max крутящий момент": ("Нм", float),
              "Max диаметр сверления (металл)": ("мм", int),
              "Max диаметр сверления (дерево)": ("мм", int),
              "Вес": ("кг", float),
              "Емкость аккумулятора": ("А.ч", float),
              "Режимы работы": (" ", str),
              "Тип двигателя": (" ", str),
              "Класс товара": (" ", str),
              "Число оборотов на холостом ходу (первая скорость) (max)": ("об/мин", int),
              "Число оборотов на холостом ходу (вторая скорость) (max)": ("об/мин", int),
              "Число оборотов на холостом ходу (первая скорость) (min)": ("об/мин", int),
              "Число оборотов на холостом ходу (вторая скорость) (min)": ("об/мин", int),
              "Размер зажимаемой оснастки (min)": ("мм", float)}

properties_names = list(properties.keys())


def parse(name, value):
    (suffix, f) = properties[name]
    if f == float:
        value = value.replace(',', '.')
        value = value.split('/')[0]
    return f(value.replace(suffix, ''))


def get_path_prop(n, k):
    return "//*[@class='product-specs__table']/table/tbody/tr[" + str(n) + "]/td[" + str(k) + "]"


def get_name_value(root, k):
    a = root.xpath(get_path_prop(k, 1))
    if not a:
        return None, None
    else:
        return a[0].text, root.xpath(get_path_prop(k, 2))[0].text


def parse_item(root):
    prop_item = len(properties) * [None]
    name = root.xpath("//*[@class='page-header__title page-header__title "
                      "page-header__title_no-uppercase']")[0].text
    code = root.xpath("//*[@class='product-header__code-title']")[0].text
    price = float(root.xpath("//*[@class='product-buy__price-value']/span")[0].text.replace("\xa0", ""))
    k = 1
    name_prop, value_prop = get_name_value(root, k)
    while name_prop:
        if name_prop in properties_names:
            prop_item[properties_names.index(name_prop)] = parse(name_prop, value_prop)
        k += 1
        name_prop, value_prop = get_name_value(root, k)
    prop_item = [name, code, price] + prop_item
    return prop_item


url = "https://spb.kuvalda.ru"
url_catalog = "/catalog/1855-shurupoverty-akkumuljatornye/page-"
number_pages = 41

with open('dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    title = ["Название", "Код товара", "Цена"] + properties_names
    writer.writerow(title)

    for i in range(1, number_pages + 1):
        page_root = l.fromstring(requests.get(url + url_catalog + str(i)).text)
        for j in range(24):
            x = page_root.xpath("//*[@id='add-content-block']/div[" + str(j + 1) + "]/div/div[2]/div[1]/a")
            if not x:
                break
            url_item = x[0].get(
                "href")
            root_item = l.fromstring(requests.get(url + url_item).text)
            writer.writerow(parse_item(root_item))

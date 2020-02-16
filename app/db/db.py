from pony.orm import *
from datetime import date
import json

db = Database()

class Tenders(db.Entity):
    item_name = Required(str)
    start_price = Required(float)
    inn = Required(str)
    kpp = Required(str)
    time_end = Required(date)
    ste_id = Required(str)
    address = Required(str)
    qnt_items = Required(int)
    price_item = Required(float)

class Items(db.Entity):
    Id = Required(int)
    name = Required(str)
    producer = Required(str)
    product_type = Required(str)
    weight = Required(str)
    offer = Required(int)
    contracts = Required(int)
    match_id = PrimaryKey(int, auto=True)

db.bind(provider='sqlite', filename='database1.sqlite')
db.generate_mapping(create_tables=True)

@db_session
def get_item(match_id):
    item = Items[match_id]
    return {"Id":item.Id,"name":item.name,"producer":item.producer,"product_type":item.product_type,"weight":item.weight,"offer":item.offer,"contracts":item.contracts}, item.name

@db_session
def get_tender(search_item):
    tenders = Tenders.select(lambda x: x.item_name == search_item)
    arr = []
    for value in tenders:
        print(value.time_end)
        arr.append({"item_name":value.item_name,"start_price":value.start_price,"inn":value.inn,"kpp":value.kpp,\
                   "time_end":value.time_end.strftime('%Y-%m-%d'),"ste_id":value.ste_id,"address":value.address,"qnt_items":value.qnt_items,\
                   "price_item":value.price_item})
    return arr

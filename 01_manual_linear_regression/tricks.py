import random

def simple_trick(b, m, r, price): # price и num_rooms - это правильные ответы
    small_random_1 = random.random()*0.1
    small_random_2 = random.random()*0.1
    p = b + m*r
    if price > p and r > 0:
        m += small_random_1
        b += small_random_2
    if price < p and r > 0:
        m -= small_random_1
        b -= small_random_2
    return m, b

def absolute_trick(b, m, r, price, learning_rate): # price и r - это правильные ответы; m, b подаем сначала рандомные
    p = b + m*r
    if price > p:
        m += learning_rate*r # мы делаем learning_rate*r, потому что r вносит вес). цена_такси = посадка + цена_за_км * километры
        b += learning_rate
    else:
        m -= learning_rate*r
        b -= learning_rate
    return m, b

def square_trick(b, m, r, price, learning_rate): # price и r - это правильные ответы; m, b подаем сначала рандомные
    p = b + m*r
    m += learning_rate*r*(price-p)
    b += learning_rate*(price-p)
    return m, b
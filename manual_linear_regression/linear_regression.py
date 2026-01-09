import random
from tricks import square_trick
from plotting import draw_line, plot_points
from rmse import rmse

random.seed(0)  # We set the random seed in order to always get the same results.

def linear_regression(x, y, learning_rate=0.01,
                      epochs=10):  # дефолтные значения у learning_rate и epochs. Теперь их не обязательно указывать
    m = random.random()
    b = random.random()
    errors = []
    for epoch in range(epochs):  # epoch = один проход обучения, общепринятое сокращение
        if epoch % 50 == 0: # Прорисовка через каждые 50 epoch. Прорисовка всех epoch: if True:
            draw_line(m, b, starting=0, ending=8)

        y_hat = x[0] * m + b # x[0] - костыль для новичка, но математически это д.б. mx+b
        errors.append(rmse(y, y_hat)) # rmse от правдивого y и предсказанного y_hat.

        i = random.randint(0, len(x) - 1)  # Нахождение одного случайного индекса
        num_rooms = x[i] # правдивое значение
        price = y[i] # правдивое значение

        m, b = square_trick(b, m, num_rooms, price, learning_rate) # подаем рандом b, рандом m, правдивые num_rooms и price
    return m, b, errors


# ===========================================
# Нахождение одного случайного индекса.
# Для этого берем длину всего списка и берем случайное значение.
# После берем уже существующее значение, но со случайным индексом [i].
# ===========================================

# ===========================================
# m, b: square_trick возвращает кортеж (return m, b). Вместо result = square_trick, result[0] и result[1] в python
# можно написать сразу "m, b = square_trick(..."
# ===========================================
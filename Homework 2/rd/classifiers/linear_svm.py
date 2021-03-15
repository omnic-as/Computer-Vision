from builtins import range
import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Функция потери SVM, наивная версия (с циклами).

    Входящие данные имеют размерность D, C классов, работаем с партиями по N примеров.

    Аргументы:
    - W: массив весов размера (D, C)
    - X: массив размера (N, D), содержит партию из N тренировочных данных
    - y: массив тренировочных меток размера (N,)
    - reg: (float) уровень регуляризации

    Возвращает:
    - потеря (float)
    - градиент функции по отношению к весам в W; массив размера (D, C)
    """
    dW = np.zeros(W.shape) # инициализируем градиент нулями

    # вычисляем общую потерю и градиент
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

    loss /= num_train

    # добавляем регуляризацию
    loss += reg * np.sum(W * W)

    ############################################################################
    # TODO:                                                                    #
    # Вычислите градиент функции потерь и сохраните его в dW.                  #
    # Вместо того, чтобы сначала вычислять потерю, а затем производную,        #
    # может быть проще вычислить производную одновременно с потерей.           #
    # Вам может потребоваться изменить код для вычисления градиента (выше).    #
    ############################################################################
    # *****НАЧАЛО ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ / ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

    pass

    # *****КОНЕЦ ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ / ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
     Функция потери SVM, векторизированная версия.
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    #############################################################################
    # TODO:                                                                     #
    # Реализуйте векторизованную версию потери SVM, сохраните результат в loss. #                                                       #
    #############################################################################
    # *****НАЧАЛО ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ / ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

    pass

    # *****КОНЕЦ ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ / ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

    #######################################################################################
    # TODO:                                                                               #
    # Реализуйте векторизованную версию градиента потери SVM, сохраните результат в dW.   #                                        #
    #######################################################################################
    # *****НАЧАЛО ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ / ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

    pass

    # *****КОНЕЦ ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ / ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

    return loss, dW
# Результаты построения модели предсказания цены автомобиля
## 1. EDA. Основные выводы:
### 1.1. Визуальный анализ
![alt text](image-1.png)
В результате визуального анализа были выявлены следующие зависимости между ценой автомобиля и его характеристиками:
* **Год выпуска**: Наблюдается прямая связь — чем новее автомобиль, тем выше его цена.
* **Пробег**: Обратная связь — с увеличением пробега цена автомобиля снижается.
* **Километраж на литр (mileage)**: Возможно, существует обратная связь — чем меньше километров автомобиль проезжает на одном литре бензина, тем ниже его стоимость. Однако данные демонстрируют значительную гетероскедастичность, и я не уверен, что эта переменная будет статистически значимой.
* **Объем двигателя, мощность и крутящий момент**: Эти параметры также имеют прямую связь с ценой — чем выше значения, тем дороже автомобиль. Однако стоит учитывать возможную коллинеарность между этими переменными. 
* **Количество посадочных мест**: Здесь наблюдается обратная связь — с увеличением числа мест цена автомобиля снижается. Это может показаться неинтуитивным: например, двуместные спортивные автомобили могут стоить дороже, чем другие модели, в то время как минивэны обычно должны иметь более высокую цену по сравнению с легковыми автомобилями. 
### 1.2. Анализ матрицы корреляций
![alt text](image.png)  
Результаты анализа коэффициентов корреляции:
* **Год выпуска и объем двигателя**: Эти переменные имеют наименьшую корреляцию (коэффициент равен 0). Это может быть связано с тем, что объем двигателей варьируется в зависимости от класса автомобиля и его характеристик, несмотря на общую тенденцию к уменьшению объемов из-за использования турбин.
* **Объем двигателя, мощность и крутящий момент**: Эти параметры сильно коррелируют между собой, что логично. Для построения модели имеет смысл использовать только один из этих факторов — например, мощность, так как она наиболее сильно связана с ценой. Это поможет избежать проблем с мультиколлинеарностью и переобучением.
* **Год выпуска и пробег**: Существует обратная связь — чем старше автомобиль, тем выше пробег (коэффициент корреляции составляет -0.37).
### 1.3. Дополнительные наблюдения
1. Интересно отметить, что новые автомобили обычно имеют меньший расход топлива (больше километров на литре). Коэффициент корреляции составляет 0.34. Это может быть связано с растущим вниманием к экологии и стремлением создавать более экономичные автомобили на фоне высоких цен на топливо.
2. Автомобили с большим количеством посадочных мест часто имеют меньшую экономию топлива. Это объясняется тем, что такие автомобили (например, компактвэны и минивэны) обычно оснащены более мощными двигателями (коэффициент корреляции между объемом двигателя и количеством мест составляет 0.65), которые потребляют больше топлива.
## 2. Результаты регрессионного анализа 
### 2.1. Простая линейная регрессия на количественных признаках
Сначала была разработана модель простой линейной регрессии, основанная исключительно на количественных признаках. Результаты показали, что наиболее значимым признаком, судя по величине бета-коэффициента, является число посадочных мест. Однако модель демонстрирует низкие коэффициенты детерминации, что делает ее применение нецелесообразным.
Для нестандартизированных данных:
* R2 (train): 54%
* R2 (test): 57%
Для стандартизированных данных:
* R2 (train scaled): 54%
* R2 (test scaled): 57%
### 2.2. Модель Lasso-регрессии
Модель строилась на количественных признаках. Также не показала прироста качества:
* R2 (train lasso): 54%
* R2 (test lasso): 57%
Ни один из бета-коэффициентов не был занулен, и даже при увеличении параметра альфа зануление не произошло. Это может свидетельствовать о том, что модель считает все признаки важными для прогноза.
Для улучшения качества модели был проведен поиск оптимального гиперпараметра alpha с помощью GridSearch и ElasticNet. Оба метода показали примерно схожие параметры alpha. Однако применение этого гиперпараметра также не привело к улучшению результатов.
![alt text](image-2.png)
### 2.3. Модель Ridge-регрессии с качественными признаками
Далее было проведено добавление качественных переменных и изменение Lasso регрессии на Ridge, что позволило изменить подход к регуляризации и обогатить датасет новой информацией.
Отмечаем, что колонка Name содержит ценную информацию от марке и модели автомобиля. Данная информация также может быть ценной для моделирования, в связи с этим можно было бы сгенерировать две новых фичи: марка и модель.
Все качественные переменные были преобразованы методом OneHot-кодирования. Поиск оптимального гиперпараметра alpha осуществлялся с помощью GridSearch с кросс-валидацией по 10 фолдам:
* Лучшие alpha: 6.5793322465756825
* Лучшее значение R^2: 0.63
Данный подход позволил значительно улучшить качество модели:
* R2 (train ridge): 66%
* R2 (test ridge): 65%
В связи с этим данная модель является наилучшей с точке зрения статистического качества среди всех ранее проанализированных.
### 2.4. Бизнес-метрики
Для оценки бизнес-метрик использовался показатель "Доля прогнозов, отличающихся от реальных цен более чем на 10% в ту или иную сторону". Расчет доли для всех выше проанализированных моделей:
* Простая линейная регрессия (без стандартизации): 82.00
* Простая линейная регрессия (со стандартизацией): 82.00
* Лассо: 81.90
* Ridge с категориальными признаками: 75.30
Наилучший результат показала Ridge-регрессия с категориальными признаками. 
### 2.5. Вывод
Модель ридж-регрессии продемонстрировала наилучшие результаты как по коэффициенту детерминации, так и по бизнес-метрикам, что делает ее рекомендованной для использования в прогнозах.
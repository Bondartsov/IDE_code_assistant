/* Файл: app.js */

// URL вашего API
const API_BASE_URL = "http://localhost:8000"; // Измените на актуальный базовый URL

let selectedEndpoint = null; // Хранит информацию о выбранном эндпоинте

// Загружаем список эндпоинтов из файла endpoints.json
fetch('endpoints.json')
    .then(response => response.json())
    .then(data => {
        // Остальной код
        const endpointsList = document.getElementById('endpoints-list');
        data.forEach(endpoint => {
            const endpointItem = document.createElement('li');
            const endpointLink = document.createElement('a');
            endpointLink.href = '#';
            endpointLink.className = 'nav-link';
            endpointLink.textContent = `${endpoint.method} ${endpoint.endpoint}`;
            endpointLink.title = endpoint.description;
            endpointLink.addEventListener('click', () => {
                // Обработка выбора эндпоинта
                selectEndpoint(endpoint);
                // Устанавливаем класс "active" для текущего элемента
                const links = endpointsList.getElementsByTagName('a');
                for (let link of links) {
                    link.classList.remove('active');
                }
                endpointLink.classList.add('active');
            });
            endpointItem.appendChild(endpointLink);
            endpointsList.appendChild(endpointItem);
        });
    })
    .catch(error => console.error('Ошибка загрузки эндпоинтов:', error));

// Функция для обработки выбора эндпоинта
function selectEndpoint(endpoint) {
    selectedEndpoint = endpoint;
    document.getElementById('endpoint-title').textContent = `${endpoint.method} ${endpoint.endpoint}`;
    document.getElementById('method').value = endpoint.method;
    document.getElementById('url').value = API_BASE_URL + endpoint.endpoint;

    // Очищаем заголовки и параметры
    document.getElementById('headers-container').innerHTML = '';
    document.getElementById('params-container').innerHTML = '';
        document.getElementById('body').value = '';
    document.getElementById('file').value = '';

    // Добавляем стандартный заголовок api-key
    addHeader('api-key', '');

    // Показать или скрыть поля в зависимости от эндпоинта
    if (endpoint.endpoint === "/api/generate_key/") {
        // Эндпоинт не требует API-ключа и дополнительных данных
        document.getElementById('body-group').style.display = 'none';
        document.getElementById('file-group').style.display = 'none';
    } else if (endpoint.endpoint === "/api/models/") {
        document.getElementById('body-group').style.display = 'none';
        document.getElementById('file-group').style.display = 'none';
    } else if (endpoint.endpoint === "/api/openai/") {
        document.getElementById('file-group').style.display = 'none';
        document.getElementById('body-group').style.display = 'block';
        document.getElementById('body').placeholder = 'Введите ваш запрос (prompt)';
    } else if (endpoint.endpoint === "/api/knowledge_base/") {
        document.getElementById('file-group').style.display = 'block';
        document.getElementById('body-group').style.display = 'block';
        document.getElementById('body').placeholder = 'Введите содержание документа (content), если необходимо';
        addParam('title', '');
    } else if (endpoint.endpoint === "/api/search/") {
        document.getElementById('file-group').style.display = 'none';
        document.getElementById('body-group').style.display = 'block';
        document.getElementById('body').placeholder = 'Введите поисковый запрос (query)';
            } else {
        // Другие эндпоинты
        document.getElementById('file-group').style.display = 'none';
        document.getElementById('body-group').style.display = 'block';
        document.getElementById('body').placeholder = '';
    }
}

// Функция для добавления заголовка
function addHeader(key = '', value = '') {
    const headersContainer = document.getElementById('headers-container');
    const headerRow = document.createElement('div');
    headerRow.className = 'form-row mb-2 header-row';

    const keyDiv = document.createElement('div');
    keyDiv.className = 'col';
    const keyInput = document.createElement('input');
    keyInput.type = 'text';
    keyInput.className = 'form-control';
    keyInput.placeholder = 'Ключ';
    keyInput.value = key;
    keyDiv.appendChild(keyInput);

    const valueDiv = document.createElement('div');
    valueDiv.className = 'col';
    const valueInput = document.createElement('input');
    valueInput.type = 'text';
    valueInput.className = 'form-control';
    valueInput.placeholder = 'Значение';
    valueInput.value = value;
    valueDiv.appendChild(valueInput);

    const removeDiv = document.createElement('div');
    removeDiv.className = 'col-1';
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn btn-danger remove-header';
    removeBtn.textContent = '×';
    removeBtn.addEventListener('click', () => {
        headerRow.remove();
    });
    removeDiv.appendChild(removeBtn);

    headerRow.appendChild(keyDiv);
    headerRow.appendChild(valueDiv);
    headerRow.appendChild(removeDiv);

    headersContainer.appendChild(headerRow);
}

// Функция для добавления параметра
function addParam(key = '', value = '') {
    const paramsContainer = document.getElementById('params-container');
    const paramRow = document.createElement('div');
    paramRow.className = 'form-row mb-2 param-row';

    const keyDiv = document.createElement('div');
    keyDiv.className = 'col';
    const keyInput = document.createElement('input');
    keyInput.type = 'text';
    keyInput.className = 'form-control';
    keyInput.placeholder = 'Ключ';
    keyInput.value = key;
    keyDiv.appendChild(keyInput);

    const valueDiv = document.createElement('div');
    valueDiv.className = 'col';
    const valueInput = document.createElement('input');
    valueInput.type = 'text';
    valueInput.className = 'form-control';
    valueInput.placeholder = 'Значение';
    valueInput.value = value;
    valueDiv.appendChild(valueInput);

    const removeDiv = document.createElement('div');
    removeDiv.className = 'col-1';
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn btn-danger remove-param';
    removeBtn.textContent = '×';
    removeBtn.addEventListener('click', () => {
        paramRow.remove();
    });
    removeDiv.appendChild(removeBtn);

    paramRow.appendChild(keyDiv);
    paramRow.appendChild(valueDiv);
    paramRow.appendChild(removeDiv);

    paramsContainer.appendChild(paramRow);
}

// Обработка добавления нового заголовка
document.getElementById('add-header').addEventListener('click', () => {
    addHeader();
});

// Обработка добавления нового параметра
document.getElementById('add-param').addEventListener('click', () => {
    addParam();
});

// Обработка отправки запроса
document.getElementById('request-form').addEventListener('submit', event => {
    event.preventDefault();
    sendRequest();
});

// Функция отправки запроса
function sendRequest() {
    const method = document.getElementById('method').value;
    let url = document.getElementById('url').value;

    // Собираем заголовки
    let headers = {};
    const headerRows = document.getElementsByClassName('header-row');
    for (let row of headerRows) {
        const key = row.children[0].children[0].value;
        const value = row.children[1].children[0].value;
        if (key) {
            headers[key] = value;
        }
    }

    // Собираем параметры запроса
    let params = {};
    const paramRows = document.getElementsByClassName('param-row');
    for (let row of paramRows) {
        const key = row.children[0].children[0].value;
        const value = row.children[1].children[0].value;
        if (key) {
            params[key] = value;
        }
    }

    // Тело запроса
    let data = null;
    const bodyContent = document.getElementById('body').value;

    // Проверяем, требуется ли отправка данных формы или JSON
    if (method === 'POST' || method === 'PUT') {
        // Проверяем, есть ли файл
        const fileInput = document.getElementById('file');
        if (fileInput && fileInput.files.length > 0) {
            // Используем FormData для отправки файла и других данных
            data = new FormData();

            // Добавляем файл
            data.append('file', fileInput.files[0]);

            // Добавляем дополнительные данные (например, title, content)
            // Предполагаем, что пользователь вводит их в параметрах запроса
                    for (let key in params) {
                        data.append(key, params[key]);
            }

            // Если есть текстовое содержание, добавляем его
                    if (bodyContent) {
                        data.append('content', bodyContent);
}

            // Заголовок 'Content-Type' будет автоматически установлен при использовании FormData
            delete headers['Content-Type'];
        } else {
            // Если нет файла, но есть данные в теле
            if (bodyContent || Object.keys(params).length > 0) {
                // Отправляем данные как форму (application/x-www-form-urlencoded)
                data = new URLSearchParams();

                // Если для эндпоинта требуется определенное поле (например, 'prompt' для /api/openai/)
                if (selectedEndpoint.endpoint === '/api/openai/') {
                    data.append('prompt', bodyContent);
                } else if (selectedEndpoint.endpoint === '/api/search/') {
                    data.append('query', bodyContent);
                } else {
                    // Добавляем поля из параметров
                    for (let key in params) {
                        data.append(key, params[key]);
        }
                    // Добавляем тело запроса, если необходимо
                    if (bodyContent) {
                        data.append('content', bodyContent);
    }
                }

                headers['Content-Type'] = 'application/x-www-form-urlencoded';
            }
        }
    }

    // Опции запроса
    const config = {
        method: method,
        url: url,
        headers: headers,
        params: method === 'GET' || method === 'DELETE' ? params : {},
        data: data ? data : null
    };

    // Логируем запрос
    console.log('Запрос:', config);

    // Отправляем запрос с помощью Axios
    axios(config)
        .then(response => {
            // Обработка успешного ответа
            console.log('Ответ:', response);
            const formattedResponse = JSON.stringify(response.data, null, 2);
            document.getElementById('response-content').textContent = formattedResponse;
        })
        .catch(error => {
            // Обработка ошибок
            if (error.response) {
                // Сервер вернул ответ с ошибкой
                console.error('Ошибка ответа:', error.response);
                const errorInfo = `Статус: ${error.response.status}\nСообщение: ${JSON.stringify(error.response.data, null, 2)}`;
                document.getElementById('response-content').textContent = errorInfo;
            } else if (error.request) {
                // Запрос был отправлен, но ответа не получено
                console.error('Ошибка запроса:', error.request);
                document.getElementById('response-content').textContent = 'Ошибка запроса. Сервер не ответил.';
            } else {
                // Другие ошибки
                console.error('Ошибка:', error.message);
                document.getElementById('response-content').textContent = 'Ошибка: ' + error.message;
            }
        });
}
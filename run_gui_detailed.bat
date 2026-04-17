@echo off
setlocal

echo ====================================================
echo Запуск приложения UniScan
echo ====================================================
echo.

REM Проверка наличия виртуального окружения
if not exist ".venv\Scripts\activate" (
    echo ОШИБКА: Виртуальное окружение не найдено!
    echo Пожалуйста, убедитесь, что папка .venv существует в корне проекта.
    echo.
    pause
    exit /b 1
)

echo Активация виртуального окружения...
call .venv\Scripts\activate

REM Проверка успешной активации
if "%VIRTUAL_ENV%"=="" (
    echo ОШИБКА: Не удалось активировать виртуальное окружение!
    echo.
    pause
    exit /b 1
)

echo Виртуальное окружение активировано успешно.
echo Путь к окружению: %VIRTUAL_ENV%
echo.

REM Проверка Python
python -c "import sys; print('Используется Python:', sys.executable)"

echo.
echo Запуск HTTP сервера для UniScan...
echo.

REM Запуск приложения
python -m uniscan serve-http

echo.
echo Сервер завершил работу.
echo.
pause
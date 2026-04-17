@echo off
echo Запуск приложения UniScan с активированным виртуальным окружением...
echo.

REM Активация виртуального окружения
call .venv\Scripts\activate

REM Проверка, что виртуальное окружение активировано
python -c "import sys; print('Используется Python:', sys.executable)"

REM Запуск GUI приложения
python -m uniscan serve-http

echo.
echo Приложение завершено.
pause
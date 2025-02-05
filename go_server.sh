if [ -n "$2" ]; then
    port=$2
else
    port=8008
fi

shiny run --reload --port "${port}" "${1}/app.py"
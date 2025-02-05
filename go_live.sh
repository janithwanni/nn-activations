if [ -n "$2" ]; then
    port=$2
else
    port=8008
fi

echo "python -m http.server --directory _${1}_site ${port}"

shinylive export $1 "_${1}_site"
python -m http.server --directory "_${1}_site" "${port}"
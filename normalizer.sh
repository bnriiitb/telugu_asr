path="$1"
ext=".m4a"
for f in $(find "$path" -type f -name "*$ext"); do 
ffmpeg -loglevel warning -hide_banner -stats -i "$f" -ar 16000 -ac 1 "${f/%m4a/wav}" && rm "$f" & done
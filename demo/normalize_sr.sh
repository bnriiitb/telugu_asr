path="$1"
ext=".m4a"
for f in $(find "$path" -type f -name "*$ext"); do 
ffmpeg -loglevel warning -hide_banner -stats -i "$f" -ar 16000 -ac 1 "${f/%m4a/wav}" && rm "$f" & 
done
# do
# # ffmpeg -loglevel warning -hide_banner -stats -i "$f" -ar 16000 -ac 1 "$f$ext" && rm "$f" && mv "$f$ext" "$f" &
# # ffmpeg -loglevel warning -hide_banner -stats -i "$f" -ar 16000 -ac 1 "${f/%m4a/wav}" && rm "$f" && mv "${f/%m4a/wav}" "$f" &

# done

# for f in *.m4a; do ffmpeg -i "$f" "${f/%m4a/wav}"; done
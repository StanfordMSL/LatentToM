#!/usr/bin/env bash
shopt -s nullglob

for f in *.mov; do
  echo "Converting $f …"
  name="${f%.*}"               # 去掉 .mov 后缀
  tmp="tmp_${name}.mp4"
  ffmpeg -y -i "$f" \
    -c:v libx264 \
    -profile:v baseline \
    -level 3.0 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -c:a aac \
    -b:a 128k \
    "$tmp"
  if [ $? -eq 0 ]; then
    mv -f "$tmp" "${name}.mp4"
    echo "  → ${name}.mp4 done"
  else
    echo "  ! failed on $f"
    rm -f "$tmp"
  fi
done

echo "All .mov converted to H.264 MP4!"


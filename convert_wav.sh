base_path="/home/tuht/tacotron1/tiengviet/wav"
out_path="/home/tuht/tacotron1/tiengviet/wav_convert"
for file in $base_path/*.wav; do
    echo $(basename "$file");
    sox $file -r 22050 -c 1 -b 16 $out_path/$(basename "$file");
    done



#!/bin/bash

mapfile -t text < input.txt
num=${text[0]}
stud=()

if [[ $num -eq 0 ]]; then
    > output.txt  
    exit 0  
fi

for ((i = 1; i <= num; i++)); do
    stud[$i]="${text[i]}"
done

approach=${text[num + 1]}
> output.txt

if [[ "$approach" = "date" ]]; then
    printf "%s\n" "${stud[@]}" | LC_ALL=C sort -k5n,5 -k4n,4 -k3n,3 -k1,1 -k2,2 | while read -r name surname day month year; do
        echo "$name $surname $day.$month.$year" >> output.txt
    done
elif [[ "$approach" = "name" ]]; then
    printf "%s\n" "${stud[@]}" | LC_ALL=C sort -k2,2 -k1,1 -k5n,5 -k4n,4 -k3n,3 | while read -r name surname day month year; do
        echo "$name $surname $day.$month.$year" >> output.txt
    done
fi

#!/bin/bash

mapfile -t numbers < "input.txt"

mod=1000000007
x=${numbers[0]}
res=0
power_of_x=1
n=${#numbers[@]}

for (( i=1; i<n; i++ ))
do
    coefficient=${numbers[$i]}
    res=$(( (res + coefficient * power_of_x) % mod ))
    power_of_x=$(( (power_of_x * x) % mod ))
done

echo $res > output.txt
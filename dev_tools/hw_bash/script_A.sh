{
    echo "Input file contains:"
    echo "$(grep -o '[[:alpha:]]' input.txt | wc -l) letters"
    echo "$(wc -w < input.txt) words"
    echo "$(grep -c '' input.txt) lines"
} > output.txt